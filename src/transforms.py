import torchvision.transforms as transforms

def get_transforms(img_size=224):
    """
    Returns data transformations for the 2-Stage Training Strategy.
    
    Returns:
        transform_stage1: Strong augmentation for representation learning.
        transform_stage2: Weak augmentation for fine-tuning.
        transform_test: Standard preprocessing for validation/testing.
    """
    
    # [Stage 1] Strong Augmentation
    # Goal: Learn general features and prevent overfitting (Representation Learning)
    transform_stage1 = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # [Stage 2] Weak Augmentation
    # Goal: Refine decision boundaries with distribution closer to ground truth (Fine-tuning)
    transform_stage2 = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Remove color distortion to match real data distribution
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # [Test/Validation] Standard Preprocessing
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return transform_stage1, transform_stage2, transform_test