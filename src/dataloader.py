import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import config


def get_transform(split="train"):
    if split == "train":
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_P),
            T.ColorJitter(**config.COLOR_JITTER),
            T.RandomRotation(degrees=config.RANDOM_ROTATION_DEG),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def scan_lfw(data_path, min_images=2):
    mapping = {}
    for person in sorted(os.listdir(data_path)):
        person_dir = os.path.join(data_path, person)
        if not os.path.isdir(person_dir):
            continue
        imgs = [os.path.join(person_dir, f)
                for f in os.listdir(person_dir) if f.endswith(".jpg")]
        if len(imgs) >= min_images:
            mapping[person] = sorted(imgs)
    return mapping

def merge_misclassified_into_train(train_map, misclassified_path):
    """
    Merge corrected face crops into train_map ONLY.
    Called after split_identities() so crops never leak into val or test.
    """
    if not misclassified_path or not os.path.exists(misclassified_path):
        print("[dataloader] No misclassified data found — training on LFW only.")
        return train_map
 
    merged_identities, merged_images = 0, 0
    for person in sorted(os.listdir(misclassified_path)):
        person_dir = os.path.join(misclassified_path, person)
        if not os.path.isdir(person_dir):
            continue
        imgs = [os.path.join(person_dir, f)
                for f in os.listdir(person_dir) if f.lower().endswith(".jpg")]
        if not imgs:
            continue
        if person in train_map:
            # Person exists in LFW train split — append corrected crops
            train_map[person] = sorted(train_map[person] + imgs)
        else:
            # New identity not in LFW at all — add to train only
            train_map[person] = sorted(imgs)
        merged_identities += 1
        merged_images += len(imgs)
 
    print(f"[dataloader] Merged {merged_images} misclassified crops "
          f"across {merged_identities} identities into train set only.")
    return train_map

def split_identities(identity_map, train_ratio, val_ratio, seed=42):
    names = list(identity_map.keys())
    random.Random(seed).shuffle(names)
    n_train = int(len(names) * train_ratio)
    n_val   = int(len(names) * val_ratio)
    return (
        {k: identity_map[k] for k in names[:n_train]},
        {k: identity_map[k] for k in names[n_train:n_train + n_val]},
        {k: identity_map[k] for k in names[n_train + n_val:]},
    )


class PKSampler(Sampler):
    """
    Yields batches of exactly batch_size samples.
    Internally uses P identities x K images each.
    batch_size must be divisible by k.
    """
    def __init__(self, labels, batch_size, k=4, n_batches=None):
        assert batch_size % k == 0, f"batch_size ({batch_size}) must be divisible by k ({k})"
        self.p = batch_size // k
        self.k = k

        self.label_to_indices = {}
        for idx, lbl in enumerate(labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)
        self.unique_labels = list(self.label_to_indices.keys())
        self.n_batches = n_batches or max(1, len(labels) // batch_size)

    def __iter__(self):
        for _ in range(self.n_batches):
            chosen = (
                random.choices(self.unique_labels, k=self.p)
                if len(self.unique_labels) < self.p
                else random.sample(self.unique_labels, self.p)
            )
            batch = []
            for lbl in chosen:
                batch.extend(random.choices(self.label_to_indices[lbl], k=self.k))
            yield batch

    def __len__(self):
        return self.n_batches


class LFWIdentityDataset(Dataset):
    def __init__(self, identity_map, transform=None):
        self.transform = transform
        self.samples   = []
        for label_idx, (_, img_paths) in enumerate(identity_map.items()):
            for p in img_paths:
                self.samples.append((p, label_idx))
        self.labels = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class LFWPairDataset(Dataset):
    def __init__(self, identity_map, n_pairs=2000, transform=None, seed=0):
        self.transform = transform
        rng   = random.Random(seed)
        names = list(identity_map.keys())
        multi = [n for n in names if len(identity_map[n]) >= 2]
        half  = n_pairs // 2

        pairs = []
        for _ in range(half):
            name = rng.choice(multi)
            a, b = rng.sample(identity_map[name], 2)
            pairs.append((a, b, 0))
        for _ in range(n_pairs - half):
            n1, n2 = rng.sample(names, 2)
            pairs.append((rng.choice(identity_map[n1]), rng.choice(identity_map[n2]), 1))

        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=64, k=4, n_test_pairs=2000):
    """
    batch_size : total images per batch (must be divisible by k)
    k          : images per identity — P is derived as batch_size // k
    """
    assert batch_size % k == 0, f"batch_size ({batch_size}) must be divisible by k ({k})"
    p = batch_size // k

    identity_map = scan_lfw(config.RAW_DIR)
    print(f"Found {len(identity_map)} identities")

    train_map, val_map, test_map = split_identities(
        identity_map, config.TRAIN_RATIO, config.VAL_RATIO, config.SEED
    )
    print(f"train={len(train_map)}  val={len(val_map)}  test={len(test_map)}")

    misclassified_path = os.path.join(config.DATA_DIR, "misclassified")
    train_map = merge_misclassified_into_train(train_map, misclassified_path)
    print(f"After merge  — train={len(train_map)}  val={len(val_map)}  test={len(test_map)}")
 

    train_ds = LFWIdentityDataset(train_map, transform=get_transform("train"))
    val_ds   = LFWIdentityDataset(val_map,   transform=get_transform("val"))
    test_ds  = LFWPairDataset(test_map, n_pairs=n_test_pairs, transform=get_transform("test"))

    common = dict(num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKSampler(train_ds.labels, batch_size=batch_size, k=k),
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=PKSampler(val_ds.labels, batch_size=batch_size, k=k,
                                n_batches=max(1, len(val_ds) // batch_size)),
        **common,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **common
    )

    print(f"Batch size: {batch_size}  (P={p} identities x K={k} images each)")
    return train_loader, val_loader, test_loader