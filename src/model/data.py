import os
import argparse
from glob import glob
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import kornia

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.environ.get(
    "PARETO_EBM_DATASET_ROOT",
    os.path.join(REPO_ROOT, "datasets"),
)

class ImageFolderNoLabel(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: directory path containing jpg/png images
        transform: torchvision transform pipeline for preprocessing
        """
        self.root_dir = root_dir
        # Collect all jpg/jpeg/png files.
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob(os.path.join(root_dir, ext)))
        self.image_paths.sort()  # Keep deterministic order for reproducibility.

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # Use a default transform when none is provided.
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),  # Target resolution.
                transforms.ToTensor(),        # [0,1], shape [C,H,W]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image and convert to RGB (source may be L/RGBA).
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
    
class GaussianBlur:

    def __init__(self, min_val=0.1, max_val=2.0, kernel_size=9):
        self.min_val = min_val
        self.max_val = max_val
        self.kernel_size = kernel_size
    
    def __call__(self, sample):
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max_val - self.min_val) * np.random.random_sample() + self.min_val
            sample = kornia.filters.GaussianBlur2d((self.kernel_size, self.kernel_size), (sigma, sigma))(sample)

        return sample

def clamp_x(x, min_val=0, max_val=1):
    return torch.clamp(x, min=min_val, max=max_val)

def enhance_buffer_data(img_shape):
    color_transform = get_color_distortion()

    return transforms.Compose([transforms.RandomResizedCrop(img_shape, scale=(0.08, 1.0)), 
                                transforms.RandomHorizontalFlip(),
                                color_transform]) 

def get_color_distortion(s=1.0):
    color_jitter = kornia.augmentation.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.4 * s)
    rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = kornia.augmentation.RandomGrayscale(p=0.2)
    color_distort = torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

# get images and labels for model training
def get_data(args):
             
    train_transforms = [
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    lambda x: (255 * x + torch.rand_like(x)) / 256.,
    lambda x: x + args.img_sigma * torch.randn_like(x),
    lambda x: clamp_x(x),
    ]

    test_transforms = [
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    lambda x: (255 * x + torch.rand_like(x)) / 256.,
    lambda x: x + args.img_sigma * torch.randn_like(x),
    lambda x: clamp_x(x),
    ]

    celeba_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT,split="all",download=True)

    indexs = []
    for idx,attr in enumerate(celeba_dataset.attr_names[:-1]):
        if len(celeba_dataset.attr[celeba_dataset.attr[:,idx]==1]) >= len(celeba_dataset) * args.celeba_drop_infreq:
            indexs.append(idx)
    label_transforms = lambda x: x[indexs]
            
    train_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="train", download=True, transform=torchvision.transforms.Compose(train_transforms),target_transform=label_transforms)
    val_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="valid", download=True, transform=torchvision.transforms.Compose(test_transforms),target_transform=label_transforms)
    test_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="test", download=True, transform=torchvision.transforms.Compose(test_transforms),target_transform=label_transforms)

    train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                shuffle=True, 
                                drop_last=True)

    valid_loader = DataLoader(dataset=val_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                shuffle=False, 
                                drop_last=False)

    test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                shuffle=False, 
                                drop_last=False)
    
    return train_loader, valid_loader, test_loader, len(indexs), indexs

# get images and labels for zero shot experiment's model training
def get_data_zero_shot(args):

    train_transforms = [
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    lambda x: (255 * x + torch.rand_like(x)) / 256.,
    lambda x: x + args.img_sigma * torch.randn_like(x),
    lambda x: clamp_x(x),
    ]

    test_transforms = [
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    lambda x: (255 * x + torch.rand_like(x)) / 256.,
    lambda x: x + args.img_sigma * torch.randn_like(x),
    lambda x: clamp_x(x),
    ]

    celeba_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT,split="all",download=True)

    indexs = []
    for idx,attr in enumerate(celeba_dataset.attr_names[:-1]):
        if len(celeba_dataset.attr[celeba_dataset.attr[:,idx]==1]) >= len(celeba_dataset) * args.celeba_drop_infreq:
            indexs.append(idx)
    label_transforms = lambda x: x[indexs]

    train_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="train", download=True, transform=torchvision.transforms.Compose(train_transforms),target_transform=label_transforms)
    val_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="valid", download=True, transform=torchvision.transforms.Compose(test_transforms),target_transform=label_transforms)
    test_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="test", download=True, transform=torchvision.transforms.Compose(test_transforms),target_transform=label_transforms)

    attr = train_dataset.attr
    names = train_dataset.attr_names

    def A(name: str):
        idx = names.index(name)
        return (attr[:, idx] == 1)

    def NOT_A(name: str):
        idx = names.index(name)
        return (attr[:, idx] == 0)

    beard        = NOT_A("No_Beard")
    male         = A("Male")
    bangs        = A("Bangs")
    blond_hair   = A("Blond_Hair")
    brown_hair   = A("Brown_Hair")
    black_hair   = A("Black_Hair")
    lipstick     = A("Wearing_Lipstick")
    heavy_makeup = A("Heavy_Makeup")

    comb1 = male & bangs            # Male & Bangs
    comb2 = male & blond_hair       # Male & Blond_Hair
    comb3 = beard & bangs           # Beard & Bangs
    comb4 = beard & brown_hair      # Beard & Brown_Hair
    comb5 = male & lipstick         # Male & Wearing_Lipstick
    comb6 = male & heavy_makeup     # Male & Heavy_Makeup
    comb7 = black_hair & blond_hair # Black_Hair & Blond_Hair

    remove_mask = comb1 | comb2 | comb3 | comb4 | comb5 | comb6 | comb7
    keep_mask = ~remove_mask

    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
    train_dataset = Subset(train_dataset, keep_indices)

    print("zero-shot samples:", len(train_dataset))
    print("number of samples deleted:", int(remove_mask.sum().item()))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, valid_loader, test_loader, len(indexs), indexs

# get image for a specific subset
def get_data_specific_subset(args):

    train_transforms = [
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    lambda x: (255 * x + torch.rand_like(x)) / 256.,
    lambda x: x + args.img_sigma * torch.randn_like(x),
    lambda x: clamp_x(x),
    ]

    test_transforms = [
    torchvision.transforms.Resize((args.img_size,args.img_size)),
    torchvision.transforms.ToTensor(),
    lambda x: (255 * x + torch.rand_like(x)) / 256.,
    lambda x: x + args.img_sigma * torch.randn_like(x),
    lambda x: clamp_x(x),
    ]

    celeba_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT,split="all",download=True)

    indexs = []
    for idx,attr in enumerate(celeba_dataset.attr_names[:-1]):
        if len(celeba_dataset.attr[celeba_dataset.attr[:,idx]==1]) >= len(celeba_dataset) * args.celeba_drop_infreq:
            indexs.append(idx)
    label_transforms = lambda x: x[indexs]

    train_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="train", download=True, transform=torchvision.transforms.Compose(train_transforms),target_transform=label_transforms)
    val_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="valid", download=True, transform=torchvision.transforms.Compose(test_transforms),target_transform=label_transforms)
    test_dataset = torchvision.datasets.CelebA(root=DATASET_ROOT, split="test", download=True, transform=torchvision.transforms.Compose(test_transforms),target_transform=label_transforms)

    train_attr = train_dataset.attr
    train_names = train_dataset.attr_names

    val_attr = val_dataset.attr
    val_names = val_dataset.attr_names

    test_attr = test_dataset.attr
    test_names = test_dataset.attr_names

    def A(names, attr, name: str):
        idx = names.index(name)
        return (attr[:, idx] == 1)

    def NOT_A(names, attr, name: str):
        idx = names.index(name)
        return (attr[:, idx] == 0)

    if args.attr_positive :
        train_subset = A(train_names, train_attr, args.attr_name)
        val_subset = A(val_names, val_attr, args.attr_name)
        test_subset = A(test_names, test_attr, args.attr_name)
    else : 
        train_subset = NOT_A(train_names, train_attr, args.attr_name)
        val_subset = NOT_A(val_names, val_attr, args.attr_name)
        test_subset = NOT_A(test_names, test_attr, args.attr_name)

    train_keep_mask = train_subset
    val_keep_mask = val_subset
    test_keep_mask = test_subset

    train_keep_indices = torch.nonzero(train_keep_mask, as_tuple=False).squeeze(1)
    val_keep_indices = torch.nonzero(val_keep_mask, as_tuple=False).squeeze(1)
    test_keep_indices = torch.nonzero(test_keep_mask, as_tuple=False).squeeze(1)

    train_dataset = Subset(train_dataset, train_keep_indices)
    val_dataset = Subset(val_dataset, val_keep_indices)
    test_dataset = Subset(test_dataset, test_keep_indices)

    print("CelebA train subset samples:", len(train_dataset))
    print("CelebA val subset samples:", len(val_dataset))
    print("CelebA test subset samples:", len(test_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, valid_loader, test_loader, len(indexs), indexs
