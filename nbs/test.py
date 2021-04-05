# load all images in a directory
from tqdm import tqdm
from os import listdir
from matplotlib import image
# load all images in a directory
loaded_images = list()
for filename in tqdm(listdir('data/train/')):
	# load image
	img_data = image.imread('data/train/' + filename)
	# store loaded image
	loaded_images.append(img_data)
	#print('> loaded %s %s' % (filename, img_data.shape))
	
	
	
images = df['filenames'].values
folders = [x for x in os.listdir('../input/training/train/')]
# Extract 9 random images from it
random_images = [np.random.choice(images) for i in range(9)]

# Location of the image dir
img_dir = '../input/training/train/' 
print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(8,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

# Adjust subplot parameters to give specified padding
plt.tight_layout()	
	
	
	
	
	
class CharDataset(Dataset):
    def __init__(self,df,im_path,transforms=None,is_test=False):
        self.df = df
        self.im_path = im_path
        self.transforms = transforms
        self.is_test = is_test
        
    def __getitem__(self,idx):
        img_path = f"{self.im_path}/{self.df.iloc[idx]['filenames']}.png"
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(**{"image": np.array(img)})["image"]
            
        if self.is_test:
            return img
        target = self.df.iloc[idx]['labels']
        return img,torch.tensor([target],dtype=torch.float32)
    
    def __len__(self):
        return self.df.shape[0]
        
        
        
        
        
        
        
        
        
        
        
