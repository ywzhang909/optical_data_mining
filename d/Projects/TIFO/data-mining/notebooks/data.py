# ... existing code ...
swanlab.init(
    # 设置项目名
    project="beam_analysis_test",
    
    # 设置超参数
    config={
        "focal_data_path": str(axis_beam_dir),
        "pupil_data_path": str(pupil_beam_dir),
        "process": '->'.join(['denoise', 'center', 'strehl']),
    }
)
# ... existing code ...