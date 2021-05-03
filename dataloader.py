import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import glob
from torch import nn as nn
from tqdm import tqdm
import torch
import torchvision
from transforms import *
from utils import Steerable_Pyramid_Phase, get_device
#import pandas as pd
class VideoRecord(object):
	def __init__(self, video, feature_dir, annot_dir, label_name, test_mode = False):
		self.video = video
		self.feature_dir = feature_dir
		self.annot_dir = annot_dir
		self.label_name = label_name
		self.test_mode = test_mode
		self.path_label = self.get_path_label()
	 
	def get_path_label(self):
		frames = glob.glob(os.path.join(self.feature_dir, self.video, '*.npy'))
		frames = sorted(frames, key	 = lambda x: os.path.basename(x).split(".")[0])
		if len(frames)==0:
			raise ValueError("number of frames of video {} should not be zero.".format(self.video))
		if '_' in self.label_name:
			self.label_name = self.label_name.split("_")
		else:
			self.label_name = [self.label_name]
		annot_file = [os.path.join(self.annot_dir, 'Training_Set', self.video+".txt") ]
		#annot_file.append(annot_file[0])
		#print(annot_file)
		if (not self.test_mode) and (any([not os.path.exists(file) for file in annot_file])):
			#cmd='rm -r ../scripts/Extracted_Features/train2/resnet50_ferplus_dag_features_fps\=30_pool5_7x7_s1/'+self.video
			#os.system(cmd)
			#print(self.video)
			raise ValueError("Annotation file not found: the training mode should always has annotation file!")
		if self.test_mode:
			return [frames, np.array([[-100] * len(self.label_name)]*len(frames))]
		else:
			total_labels = []
			for file in annot_file:
				f = open(file, "r")
				lines = f.readlines()
				corr_frames = sorted(glob.glob(os.path.join(self.feature_dir, self.video, '*.npy')))
				lines = lines[1:] # skip first line
				lines = [x.strip() for x in lines]
				if self.label_name[0]=='valence':
					lines = [float(x.split(',')[0]) for x in lines]
				else:
					lines = [float(x.split(',')[1]) for x in lines]
				#lines = [[float(y) for y in x ] for x in lines]
				f.close()
				#print(lines)
				assert len(corr_frames) <= len(lines)
				frames_ids = [int(frame.split('/')[-1].split('.')[0]) - 1 for frame in corr_frames] # frame_id start from 0
				label_array=np.array(lines)
				N = label_array.shape[0]
				label_array = label_array.reshape((N, -1))
				to_drop = (label_array == -5).sum(-1)
				drop_ids = [i for i in range(len(to_drop)) if to_drop[i]]
				frames_ids = [i for i in frames_ids if i not in drop_ids]
				indexes = [True if i in frames_ids else False for i in range(len(label_array)) ]
				label_array = label_array[indexes]
				assert len(label_array) == len(frames_ids)
				prefix = '/'.join(corr_frames[0].split('/')[:-1])
				return_frames = [prefix+'/{0:05d}.npy'.format(id+1) for id in frames_ids]
				#print(label_array)
				label_array=np.reshape(label_array, -1)
				#print(label_array)
				total_labels.append(label_array.tolist())
				#print(total_labels)
			
			total_labels = np.asarray(total_labels)
			#np.reshape(total_labels, (2,-1)) 
			#print(total_labels)
			total_labels = total_labels.transpose(1, 0)
			'''if self.video!='189':
				print(self.video)
				print(return_frames)
				print(total_labels.shape[0])
				new_df = pd.DataFrame(total_labels)
				new_df.to_csv('out.csv', index=False, header=None)
				exit()'''
			return [return_frames, total_labels]
	def __str__(self):
		string = ''
		for key, record in self.utterance_dict.items():
			string += str(record)+'\n'
		return string
def phase_2_output( phase_batch, steerable_pyramid,return_phase=False):
	"""
	phase_batch dim: bs, num_phase, W, H
	"""
	sp = steerable_pyramid
	num_frames,num_phases, W, H = phase_batch.size()
	coeff_batch = sp.build_pyramid(phase_batch)
	assert isinstance(coeff_batch, list)
	phase_batch_0 = sp.extract_phase(coeff_batch[0], return_phase=return_phase)
	num_frames, n_ch, n_ph, W, H= phase_batch_0.size()
	phase_batch_0 = phase_batch_0.view(num_frames, -1, W, H)
	phase_batch_1 = sp.extract_phase(coeff_batch[1], return_phase=return_phase)
	num_frames, n_ch, n_ph, W, H= phase_batch_1.size()
	phase_batch_1 = phase_batch_1.view(num_frames, -1, W, H)
	return phase_batch_0,phase_batch_1
class Face_Dataset(data.Dataset):
	def __init__(self, root_path, feature_path, annot_dir, video_name_list,	 label_name, py_level=4, py_nbands=2, 
				 test_mode =False, num_phase=12, phase_size = 48, length=64, stride=32, return_phase=False):
		self.root_path = root_path
		self.feature_path = feature_path
		self.annot_dir = annot_dir
		self.video_name_list = video_name_list
		self.label_name = label_name
		self.test_mode = test_mode
		self.length = length # length of sequence as input to the RNN
		self.stride = stride # 
		self.num_phase = num_phase
		self.phase_size = phase_size
		self.return_phase = return_phase
		device = get_device('cuda:0')
		self.steerable_pyramid = Steerable_Pyramid_Phase(height=py_level, nbands=py_nbands, scale_factor=2, device=device, extract_level=[1,2], visualize=False)
		print("sample stride {} is only applicable when test_mode=False.".format(stride))
		self.parse_videos()

	def parse_videos(self):
		videos = self.video_name_list	   
		self.video_list = list()
		self.sequence_ranges = []
		self.video_ids = []
		self.total_labels = []
		vid_ids =0
		for vid in tqdm(videos):
			v_record = VideoRecord(vid, self.feature_path, self.annot_dir, self.label_name, self.test_mode)
			frames, labels = v_record.path_label
			self.total_labels.append(labels)
			if len(frames) !=0 and (len(frames)==len(labels)):
				self.video_list.append(v_record)
				if self.test_mode:
					n_seq = len(frames)//self.length
					if len(frames)%self.length !=0:
						n_seq +=1
					seq_range = []
					for i in range(n_seq):
						if (i+1)*self.length<=len(frames):
							seq_range.append([i*self.length, (i+1)*self.length] )
						else:
							seq_range.append([len(frames)-self.length, len(frames)])
					self.sequence_ranges.extend(seq_range)
					self.video_ids.extend([vid_ids]*n_seq)
					vid_ids +=1
				else:
					n_seq = 0
					start, end = 0, self.length
					seq_range = []
					while end < len(frames) and (start<len(frames)):
						seq_range.append([start, end])
						n_seq +=1
						start +=self.stride
						end = start+self.length
					self.sequence_ranges.extend(seq_range)
					self.video_ids.extend([vid_ids]*n_seq) 
					vid_ids +=1
		self.total_labels = np.concatenate(self.total_labels, axis=0)				 
		print("number of videos:{}, number of seqs:{}".format(len(self.video_list), len(self)))

	def __len__(self):
		return len(self.sequence_ranges)
	def __getitem__(self, index):
		seq_ranges = self.sequence_ranges[index]
		start, end = seq_ranges
		video_record = self.video_list[self.video_ids[index]]
		frames, labels = video_record.path_label
		seq_frames, seq_labels = frames[start:end], labels[start:end]
		imgs = []
		for f in seq_frames:
			imgs.append(np.load(f))
		# phase image sample
		sample_f_ids = []
		for f_id in range(start, end):
			phase_ids = []
			for i in range(self.num_phase+1):
				step = i-self.num_phase//2
				id_0 = max(0,f_id + step)
				id_0 = min(id_0, len(frames)-1) 
				phase_ids.append(id_0)
			sample_f_ids.append(phase_ids)
		sample_frames = [[frames[id] for id in ids] for ids in sample_f_ids]
		phase_images= []
		for frames in sample_frames:
			phase_img_list = []
			for frame in frames:
				f_index = int(os.path.basename(frame).split(".")[0])
				img_frame = os.path.join(self.root_path, video_record.video, '{:05d}.jpg'.format(f_index))
				try:
				   img = Image.open(img_frame).convert('L')
				except:
					raise ValueError("incorrect face path")	   
				phase_img_list.append(img)
			phase_images.append(phase_img_list)
		if not self.test_mode:
			random_seed = np.random.randint(250)
			phase_transform = torchvision.transforms.Compose([GroupRandomHorizontalFlip(seed=random_seed),
								   GroupRandomCrop(size=int(self.phase_size*0.85), seed=random_seed),
								   GroupScale(size=self.phase_size),
								   Stack(),
								   ToTorchFormatTensor()])
		else:
			phase_transform = torchvision.transforms.Compose([
								   GroupScale(size=self.phase_size),
								   Stack(),
								   ToTorchFormatTensor()]) 
		flat_phase_images = []
		for sublist in phase_images:
			flat_phase_images.extend(sublist)
		flat_phase_images = phase_transform(flat_phase_images)
		phase_images = flat_phase_images.view(len(phase_images), self.num_phase+1, self.phase_size, self.phase_size)
		phase_images = phase_images.type('torch.FloatTensor').cuda()
		phase_batch_0,phase_batch_1 = phase_2_output( phase_images, self.steerable_pyramid, return_phase=self.return_phase)
		
		return [[phase_batch_0,phase_batch_1], np.array(imgs), np.array(seq_labels), np.array([start, end]), video_record.video]
	

if __name__ == '__main__':
	root_path = '/media/newssd/Aff-Wild_experiments/Aligned_Faces_train'
	feature_path = '/media/newssd/Aff-Wild_experiments/Extracted_Features/Aff_wild_train/resnet50_ferplus_features_fps=30_pool5_7x7_s1'
	annot_dir = '/media/newssd/Aff-Wild_experiments/annotations'
	video_names = os.listdir(feature_path)[:25]

	train_dataset = Face_Dataset(root_path, feature_path, annot_dir, video_names, label_name='arousal_valence',	 num_phase=12 , phase_size=48, test_mode=True)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, 
		batch_size = 4, 
		num_workers=0, pin_memory=False )
	for phase_f, rgb_f, label, seq_range, video_names in train_loader:
		phase_0, phase_1 = phase_f
		