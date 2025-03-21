from synchformer.synchformer_model import SynchformerModel

if __name__ == "__main__":
    model = SynchformerModel(exp_name='24-01-04T16-39-21', device='cuda:0')
    model.predict_from_path('/root/streaming-inference/video.mp4', v_start_i_sec=0.0, offset_sec=0.0)