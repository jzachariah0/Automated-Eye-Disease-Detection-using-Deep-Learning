train_dir = 'data/train'
test_dir = 'data/test'

train_disease_dir = os.path.join(train_dir, 'disease')
train_normal_dir = os.path.join(train_dir, 'normal')
test_disease_dir = os.path.join(test_dir, 'disease')
test_normal_dir = os.path.join(test_dir, 'normal')

# Function to load and resize images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
    return images

# Load and visualize some sample images
train_disease_imgs = load_images_from_folder(train_disease_dir)
train_normal_imgs = load_images_from_folder(train_normal_dir)
test_disease_imgs = load_images_from_folder(test_disease_dir)
test_normal_imgs = load_images_from_folder(test_normal_dir)

fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes[0,0].imshow(train_disease_imgs[0])
axes[0,0].set_title('Disease Train Image')
axes[0,1].imshow(train_normal_imgs[0])
axes[0,1].set_title('Normal Train Image')
axes[1,0].imshow(test_disease_imgs[0])
axes[1,0].set_title('Disease Test Image')
axes[1,1].imshow(test_normal_imgs[0])
axes[1,1].set_title('Normal Test Image')
