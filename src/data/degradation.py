import numpy as np
import cv2
import os

class Degradation():
    '''
    A class with only static methods to generate a new domain shifted dataset from a source dataset
    Hyperparameters for tuning domain shift: scale (int) - scaling factor to scale image dimensions
                                             sigma (int) - std of gaussian distributed noise (higher = more noise/corrpution)
    '''
    @staticmethod
    def nearest_neighbor_gaussian_noise(img, sigma=0, scale=0.5):
        '''
        Returns an image that has been downsampled then upsampled using nearest neighbor kernel.
        sigma=0 behaves like normal nearest neighbor kernel with 0 noise added

        param img: (np.ndarray) original source image
        param scale: (int) scaling factor to multiply source image height and width by
        return: (np.ndarray) domain shifted image afer downsampling then upsampling back to original dimensions
        '''
        h, w = img.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        small = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        noisy = restored + np.random.normal(0, sigma, restored.shape) # add gaussian distributed noise to simulate sensor noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    
    @staticmethod
    def bilinear_gaussian_noise(img, sigma=0, scale=0.5):
        '''
        Returns an image that has been downsampled then upsampled using bilinear kernel + gaussian noise.
        sigma=0 behaves like normal bilinear kernel with 0 noise added

        param img: (np.ndarray) original source image
        param sigma: (int) the std of the guassian distributed noise
        param scale: (int) scaling factor to multiply source image height and width by
        return: (np.ndarray) domain shifted image afer downsampling then upsampling back to original dimensions + adding gaussian noise
        '''
        h, w = img.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        small = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        noisy = restored + np.random.normal(0, sigma, restored.shape) # add gaussian distributed noise to simulate sensor noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def make_new_domain(data_in, data_out, sigma=1, scale=0.5, kernel='bilinear_gaussian_noise'):
        '''
        Creates a new domain shifted dataset for a single directory 

        param data_in: (str) root directory path of the source dataset
        param data_out: (str) root directory path of the new dataset
        param sigma: (int) the std of the guassian distributed noise (used with bilinear_gaussian_noise kernel)
        param scale: (int) scaling factor to multiply source image height and width by
        param kernel: (str) type of kernel to use for interpolation ( 'nearest_neighbor_guassian_noise', 'bilinear_gaussian_noise')
        return: None
        '''
        if kernel not in ('nearest_neighbor_gaussian_noise', 'bilinear_gaussian_noise'):
            print("Invalid kernel type")
            return
        data_out = os.path.join(data_out, f"{kernel}_scale_{scale}_sigma_{sigma}")
        os.makedirs(data_out, exist_ok=True)

        # train
        train_dir = os.path.join(data_in, 'train')
        img_dir = os.path.join(train_dir, 'images')
        label_dir = os.path.join(train_dir, 'labels')

        train_dir_out = os.path.join(data_out, 'train')
        os.makedirs(train_dir_out,exist_ok=True)
        img_dir_out = os.path.join(train_dir_out, 'images')
        os.makedirs(img_dir_out,exist_ok=True)
        label_dir_out = os.path.join(train_dir_out, 'labels')
        os.makedirs(label_dir_out,exist_ok=True)
        # train/images
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = os.path.join(img_dir, img_name)
            img = cv2.imread(img)
            if kernel=='bilinear_gaussian_noise':
                new_img= Degradation.bilinear_gaussian_noise(img,sigma,scale)
            else:
                new_img = Degradation.nearest_neighbor_gaussian_noise(img,sigma,scale)
            out_path = os.path.join(img_dir_out, img_name)
            cv2.imwrite(out_path, new_img)

        # train/labels
        for label_name in os.listdir(label_dir):
            if not label_name.lower().endswith(('.txt')):
                continue
            label_in = os.path.join(label_dir, label_name)
            label_out = os.path.join(label_dir_out, label_name)

            try:
                with open(label_in, 'r') as source_file, \
                    open(label_out, 'w') as destination_file:
                    for line in source_file:
                        destination_file.write(line)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(e)
                continue


        # valid
        val_dir = os.path.join(data_in, 'valid')
        img_dir = os.path.join(val_dir, 'images')
        label_dir = os.path.join(val_dir, 'labels')

        val_dir_out = os.path.join(data_out, 'valid')
        os.makedirs(val_dir_out,exist_ok=True)
        img_dir_out = os.path.join(val_dir_out, 'images')
        os.makedirs(img_dir_out,exist_ok=True)
        label_dir_out = os.path.join(val_dir_out, 'labels')
        os.makedirs(label_dir_out,exist_ok=True)
        # valid/images
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = os.path.join(img_dir, img_name)
            img = cv2.imread(img)
            if kernel=='bilinear_gaussian_noise':
                new_img= Degradation.bilinear_gaussian_noise(img,sigma,scale)
            else:
                new_img = Degradation.nearest_neighbor_gaussian_noise(img,sigma,scale)
            out_path = os.path.join(img_dir_out, img_name)
            cv2.imwrite(out_path, new_img)

        # valid/labels
        for label_name in os.listdir(label_dir):
            if not label_name.lower().endswith(('.txt')):
                continue
            label_in = os.path.join(label_dir, label_name)
            label_out = os.path.join(label_dir_out, label_name)

            try:
                with open(label_in, 'r') as source_file, \
                    open(label_out, 'w') as destination_file:
                    for line in source_file:
                        destination_file.write(line)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(e)
                continue


        # test
        test_dir = os.path.join(data_in, 'test')
        img_dir = os.path.join(test_dir, 'images')
        label_dir = os.path.join(test_dir, 'labels')

        test_dir_out = os.path.join(data_out, 'test')
        os.makedirs(test_dir_out,exist_ok=True)
        img_dir_out = os.path.join(test_dir_out, 'images')
        os.makedirs(img_dir_out,exist_ok=True)
        label_dir_out = os.path.join(test_dir_out, 'labels')
        os.makedirs(label_dir_out,exist_ok=True)
        # test/images
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = os.path.join(img_dir, img_name)
            img = cv2.imread(img)
            if kernel=='bilinear_gaussian_noise':
                new_img= Degradation.bilinear_gaussian_noise(img,sigma,scale)
            else:
                new_img = Degradation.nearest_neighbor_gaussian_noise(img,sigma,scale)
            out_path = os.path.join(img_dir_out, img_name)
            cv2.imwrite(out_path, new_img)

        # test/labels
        for label_name in os.listdir(label_dir):
            if not label_name.lower().endswith(('.txt')):
                continue
            label_in = os.path.join(label_dir, label_name)
            label_out = os.path.join(label_dir_out, label_name)

            try:
                with open(label_in, 'r') as source_file, \
                    open(label_out, 'w') as destination_file:
                    for line in source_file:
                        destination_file.write(line)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(e)
                continue

        # data.yaml
        yaml_in = os.path.join(data_in, 'data.yaml' )
        yaml_out = os.path.join(data_out, 'data.yaml')
        delimeter = "roboflow"
        try:
            with open(yaml_in, 'r') as source, \
                open(yaml_out, 'w') as dest:
                for line in source:
                    if delimeter in line:
                        break
                    dest.write(line)
        except FileNotFoundError:
            print(f"No .yaml file found at {yaml_in}")
        except Exception as e:
            print(e)


        print(f"Degraded domain created at: {data_out}")
        return data_out


#def main():
    #data_in = "/home/alan/Documents/OMSCS/CS_7643/ALPR/src/data/source_data"
    #data_out = "/home/alan/Documents/OMSCS/CS_7643/ALPR/src/data/test_data"
    #Degradation.make_new_domain(data_in, data_out, sigma=30, scale=0.1, kernel='bilinear_gaussian_noise')


#if __name__ == "__main__":
    #main()


            

