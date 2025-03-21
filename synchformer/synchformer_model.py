import torch
import torchaudio
import torchvision
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from synchformer.dataset.dataset_utils import get_video_and_audio
from synchformer.dataset.transforms import make_class_grid, quantize_offset
from synchformer.utils.utils import check_if_file_exists_else_download, which_ffmpeg
from synchformer.scripts.train_utils import get_model, get_transforms, prepare_inputs


class SynchformerModel:
    """
    A class for audio-visual synchronization prediction using the Synchformer model.
    """
    
    def __init__(self, exp_name='24-01-04T16-39-21', device='cuda:0'):
        """
        Initialize the Synchformer model.
        
        Args:
            exp_name (str): Experiment name in format 'xx-xx-xxTxx-xx-xx'
            device (str): Device to run the model on ('cuda:0', 'cpu', etc.)
        """
        self.exp_name = exp_name
        self.device = torch.device(device)
        self.vfps = 25
        self.afps = 16000
        self.in_size = 256
        
        # Load model configuration and weights
        self._load_model()
        
    def _load_model(self):
        """Load and initialize the model from configuration and checkpoint."""
        cfg_path = f'./logs/sync_models/{self.exp_name}/cfg-{self.exp_name}.yaml'
        ckpt_path = f'./logs/sync_models/{self.exp_name}/{self.exp_name}.pt'
        
        # Download model files if they don't exist
        check_if_file_exists_else_download(cfg_path)
        check_if_file_exists_else_download(ckpt_path)
        
        # Load and patch config
        self.cfg = OmegaConf.load(cfg_path)
        self.cfg = self._patch_config(self.cfg)
        
        # Initialize model
        _, self.model = get_model(self.cfg, self.device)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        
        # Create offset grid
        max_off_sec = self.cfg.data.max_off_sec
        num_cls = self.cfg.model.params.transformer.params.off_head_cfg.params.out_features
        self.grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
        
        # Get transforms
        self.transforms = get_transforms(self.cfg, ['test'])['test']
    
    def _patch_config(self, cfg):
        """Patch the configuration for compatibility with older checkpoints."""
        # The FE ckpts are already in the model ckpt
        cfg.model.params.afeat_extractor.params.ckpt_path = None
        cfg.model.params.vfeat_extractor.params.ckpt_path = None
        # Old checkpoints have different names
        cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
        return cfg
    
    def reencode_video(self, path):
        """
        Reencode video to the required format.
        
        Args:
            path (str): Path to the video file
            
        Returns:
            str: Path to the reencoded video
        """
        assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
        new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{self.vfps}fps_{self.in_size}side_{self.afps}hz.mp4'
        new_path.parent.mkdir(exist_ok=True)
        new_path = str(new_path)
        
        # Reencode video
        cmd = f'{which_ffmpeg()}'
        cmd += ' -hide_banner -loglevel panic'
        cmd += f' -y -i {path}'
        cmd += f" -vf fps={self.vfps},scale=iw*{self.in_size}/'min(iw,ih)':ih*{self.in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
        cmd += f" -ar {self.afps}"
        cmd += f' {new_path}'
        subprocess.call(cmd.split())
        
        # Extract audio as WAV
        cmd = f'{which_ffmpeg()}'
        cmd += ' -hide_banner -loglevel panic'
        cmd += f' -y -i {new_path}'
        cmd += f' -acodec pcm_s16le -ac 1'
        cmd += f' {new_path.replace(".mp4", ".wav")}'
        subprocess.call(cmd.split())
        
        return new_path
    
    def check_video_format(self, vid_path):
        """
        Check if the video has the correct format and reencode if necessary.
        
        Args:
            vid_path (str): Path to the video file
            
        Returns:
            str: Path to the (possibly reencoded) video
        """
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        
        if info['video_fps'] != self.vfps or info['audio_fps'] != self.afps or min(H, W) != self.in_size:
            print(f'Reencoding. vfps: {info["video_fps"]} -> {self.vfps};', end=' ')
            print(f'afps: {info["audio_fps"]} -> {self.afps};', end=' ')
            print(f'{(H, W)} -> min(H, W)={self.in_size}')
            return self.reencode_video(vid_path)
        else:
            print(f'Skipping reencoding. vfps: {info["video_fps"]}; afps: {info["audio_fps"]}; min(H, W)={self.in_size}')
            return vid_path
    
    def predict_from_path(self, vid_path, v_start_i_sec=0.0, offset_sec=0.0):
        """
        Predict audio-visual synchronization from a video file.
        
        Args:
            vid_path (str): Path to the video file
            v_start_i_sec (float): Start time of the video clip in seconds
            offset_sec (float): Ground truth offset in seconds for evaluation
            
        Returns:
            tuple: (probabilities, predictions, ground_truth_label)
        """
        # Check and possibly reencode video
        vid_path = self.check_video_format(vid_path)
        
        # Load video and audio
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)
        
        return self.predict(rgb, audio, meta, vid_path, v_start_i_sec, offset_sec)
    
    def predict(self, rgb, audio, meta, vid_path=None, v_start_i_sec=0.0, offset_sec=0.0):
        """
        Predict audio-visual synchronization from loaded video and audio.
        
        Args:
            rgb (torch.Tensor): Video tensor of shape (Tv, 3, H, W)
            audio (torch.Tensor): Audio tensor of shape (Ta,)
            meta (dict): Metadata dictionary
            vid_path (str, optional): Path to the video file (for reference)
            v_start_i_sec (float): Start time of the video clip in seconds
            offset_sec (float): Ground truth offset in seconds for evaluation
            
        Returns:
            tuple: (probabilities, predictions, ground_truth_label)
        """
        # Create item for transforms
        item = dict(
            video=rgb, audio=audio, meta=meta, path=vid_path or "unknown", split='test',
            targets={'v_start_i_sec': v_start_i_sec, 'offset_sec': offset_sec},
        )
        
        # Check if offset is within the trained grid
        if not (min(self.grid) <= item['targets']['offset_sec'] <= max(self.grid)):
            print(f'WARNING: offset_sec={item["targets"]["offset_sec"]} is outside the trained grid: {self.grid}')
        
        # Apply transforms
        item = self.transforms(item)
        
        # Prepare inputs for inference
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, self.device)
        
        # Forward pass
        with torch.set_grad_enabled(False):
            with torch.autocast('cuda', enabled=self.cfg.training.use_half_precision):
                _, logits = self.model(vid, aud)
        
        # Process results
        return self._process_prediction(logits, item)
    
    def _process_prediction(self, off_logits, item):
        """
        Process the model prediction.
        
        Args:
            off_logits (torch.Tensor): Logits from the model
            item (dict): Item dictionary with targets
            
        Returns:
            tuple: (probabilities, predictions, ground_truth_label)
        """
        label = item['targets']['offset_label'].item()
        print('Ground Truth offset (sec):', f'{label:.2f} ({quantize_offset(self.grid, label)[-1].item()})')
        print()
        print('Prediction Results:')
        
        off_probs = torch.softmax(off_logits, dim=-1)
        k = min(off_probs.shape[-1], 5)
        topk_logits, topk_preds = torch.topk(off_logits, k)
        
        # Remove batch dimension
        assert len(topk_logits) == 1, 'batch is larger than 1'
        topk_logits = topk_logits[0]
        topk_preds = topk_preds[0]
        off_logits = off_logits[0]
        off_probs = off_probs[0]
        
        for target_hat in topk_preds:
            print(f'p={off_probs[target_hat]:.4f} ({off_logits[target_hat]:.4f}), "{self.grid[target_hat]:.2f}" ({target_hat})')
        
        return off_probs, topk_preds, label
