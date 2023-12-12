import os
import os.path as osp
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import glob
import random

import numpy as np
import dataset.base_dataset as base_dataset
import dataset.util.bvh as bvh_util
import dataset.util.geo as geo_util
import dataset.util.plot as plot_util

from transformers import AutoModel, AutoTokenizer

class STYLE100CLIP(base_dataset.BaseMotionData):
    NAME = 'STYLE100CLIP'
    def __init__(self, config):
        #assert "use_cond" in config["model_hyperparam"] and config["model_hyperparam"]["use_cond"] == True
        self.tokenizer = AutoTokenizer.from_pretrained(config["data"]["clip_model_path"])
        self.text_model = AutoModel.from_pretrained(config["data"]["clip_model_path"])

        self.text_model.training = False
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.data_path = config["data"]["path"]     
        #[path for path in glob.glob(osp.join(self.data_path,'*/*.bvh')) if '.' not in path]
        self.class_dict = {self.get_fullstyle_label(fname):idx for idx, fname in enumerate(self.get_motion_fpaths())}
        # {(style, substyle): class_index}
        self.class_texts = [self.get_fullstyle_text(style, substyle) for (style, substyle) in  self.class_dict.keys()]
        # [style_text for i in class_index]
        self.clip_embedding = self.get_clip_class_embedding(self.class_texts)
        # [style_text_emb for i in class_index]
        super().__init__(config)

    def get_fullstyle_label(self, path):
        style = osp.dirname(path).split('/')[-1]
        substyle = osp.basename(path).split('.')[0].split('_')[1]
        return style, substyle

    def get_style_text(self, style):
        style_text = style
        return style_text
    
    def get_substyle_text(self, substyle):
        checktable = {
            "BR": ["running backwards","moving quickly backwards","going reversed in a fast speed"],
            "BW": ["walking backwards","moving in a normal speed backwards", "going slowing backwards", "going backwards walking"],
            "FR": ["run ahead", "running forwards", "moving fast onwards", "going forward quickly"],
            "FW": ["walking forwards","moving in a normal pace forwards", "going slowly towards the front"],
            "ID": ["stay put","idling","remains still","not moving","freeze motion", "default single pose","default pose"],
            "SR": ["running sideway","sidestep running", "going side way with high velocity","Go sideways while running"],
            "SW": ["sideway walking", "moving with sidestep with normal speed"],
            "TR1": [""],
            "TR2": [""],
            "TR3": [""]
        }
        substyle_text = random.choice(checktable[substyle])
        return substyle_text

    def get_fullstyle_text(self, style, substyle):
        style_text = self.get_style_text(style) 
        substyle_text = self.get_substyle_text(substyle)
        text = style_text + ',' + substyle_text 
        return text.strip(',')

    def get_clip_class_embedding(self, texts, outformat='np'):
        assert outformat in ['np','pt']
        text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=25,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        # split into max length Clip can handle
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            text_input_ids = text_input_ids[:, :self.tokenizer.
                                                model_max_length] 

        text_embeddings = self.text_model.get_text_features(
                text_input_ids.to(self.text_model.device))
        text_embeddings = text_embeddings.detach()
        if outformat == 'np':
            return text_embeddings.cpu().numpy()
        return text_embeddings

    def get_motion_fpaths(self):
        path =  osp.join(self.data_path,'**/*.{}'.format('bvh'))
        file_lst = glob.glob(path, recursive = True)
        return file_lst

    def process_label(self, fname):
        return self.tmp_labels 

    def process_data(self, fname):
        style, substyle = self.get_fullstyle_label(fname)
        class_idx = self.class_dict[(style, substyle)]
        
        # read a single file, convert them into single format
        final_x = bvh_util.read_bvh(fname, foot_idx_lst=self.foot_idx, root_idx=self.root_idx, \
            unit=self.unit, source_fps=60, target_fps=self.fps)
        if self.data_trim_begin:
            final_x = final_x[self.data_trim_begin:]
        if self.data_trim_end:
            final_x = final_x[:-self.data_trim_end]

        self.tmp_labels = np.array([class_idx for _ in range(final_x.shape[0])])

        #xs = self.x_to_jnts(final_x)
        #print(labal_text, fname, label_idx)
        #plot_util.plot_lafan1(xs, self.links, fps=self.fps)
        return final_x
    
    def x_to_jnts(self, x):
        dxdy = x[...,:2] 
        dr = x[...,2]
        jnts = np.reshape(x[...,3:3+3*self.num_jnt],(-1,self.num_jnt,3))
        dpm = np.array([[0.0,0.0,0.0]])
        dpm_lst = np.zeros((dxdy.shape[0],3))
        yaws = np.cumsum(dr)
        yaws = yaws - (yaws//(np.pi*2))*(np.pi*2)
        for i in range(1, jnts.shape[0]):
           cur_pos = np.zeros((1,3))
           cur_pos[0,0] = dxdy[i,0]
           cur_pos[0,2] = dxdy[i,1]
           dpm += np.dot(cur_pos,geo_util.rot_yaw(yaws[i]))
           dpm_lst[i,:] = copy.deepcopy(dpm)
           jnts[i,:,:] = np.dot(jnts[i,:,:], geo_util.rot_yaw(yaws[i])) + copy.deepcopy(dpm)
        return jnts
    
    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        idx_ = self.valid_idx[idx]
        motion = self.motion_flattened[idx_:idx_+self.rollout]
        class_idx = self.labels[idx_:idx_+self.rollout] 
        label = self.clip_embedding[class_idx]

        #print(motion.shape, label.shape, class_idx, idx_, self.motion_flattened.shape, self.labels.shape, self.clip_embedding.shape)
        return  motion, label
    
    def plot_jnts(self, x, path=None):
        return plot_util.plot_style100(x, self.links, self.fps, path)
    


