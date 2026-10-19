"""Production entrypoint invariants: color, scores, routing and bounded conversion."""
import argparse
import importlib.util
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"fdim"))
from dist.deep_model import Model
from dist import test_video_gen as video
from run_fdim import run_model
from utils.reader import Reader, SampledReader
from utils.convert_color import vmaf_inputs
from vmaf.calc_vmaf import calc_vmaf


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    directory = tmp_path_factory.mktemp("official-inference")
    rng = np.random.default_rng(31)
    ref = rng.integers(16, 236, (11, 64*64*3//2), dtype=np.uint8)
    dis = np.clip(ref.astype(np.int16)+rng.integers(-8, 9, ref.shape), 16, 235).astype(np.uint8)
    ref.tofile(directory/"ref.yuv")
    dis.tofile(directory/"dis.yuv")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", "64x64", "-r", "2", "-i", str(directory/"dis.yuv"), "-c:v", "ffv1",
        "-colorspace", "bt709", "-color_range", "pc", "-color_trc", "bt709",
        "-color_primaries", "bt709", str(directory/"tagged.mkv")], check=True)
    return directory


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    result = Model().eval()
    state = torch.load(ROOT/"fdim/dist/checkpoints/dist_5.0.0.ckpt", map_location="cpu", weights_only=True)
    result.load_state_dict(state.get("state_dict", state), strict=True)
    return result


def options():
    return dict(ref_fmt="yuv420p", dis_fmt="yuv420p", ref_width=64, ref_height=64,
                dis_width=64, dis_height=64, ref_bit_depth=8, dis_bit_depth=8, calc_interval=2)


@pytest.mark.parametrize("name,fmt", [("dis.yuv", "yuv420p"), ("tagged.mkv", "rgb")])
def test_sampled_pixels_match_original_color_path(media, name, fmt):
    source = Reader(str(media/name),64,64,fmt)
    expected=[]
    try:
        for index in range(11):
            ok, frame = source.next()
            assert ok
            if index%2==0:
                expected.append(frame.copy())
    finally:
        source.close()
    source = SampledReader(media/name,64,64,fmt,interval=2)
    try:
        for frame in expected:
            ok, actual = source.next()
            assert ok and np.array_equal(frame,actual)
        assert not source.next()[0]
    finally:
        source.close()


@pytest.mark.parametrize("name,fmt", [("dis.yuv", "yuv420p"), ("tagged.mkv", "rgb")])
@pytest.mark.parametrize("prefetch,pipeline", [(0,False), (1,True)])
def test_production_deep_default_is_exact(media, model, name, fmt, prefetch, pipeline):
    kwargs=options();kwargs["dis_fmt"]=fmt
    expected, before=video.evaluate_video_gen(media/"ref.yuv",media/name,model,"cpu",
        inference_mode="legacy",return_frame_scores=True,**kwargs)
    score, after=video.evaluate_video_gen(media/"ref.yuv",media/name,model,"cpu",
        prefetch=prefetch,pipeline=pipeline,return_frame_scores=True,**kwargs)
    assert after["inference_mode"]=="stream"
    assert after["frame_indices"]==[0,2,4,6,8,10]
    assert after["raw_scores"]==before["raw_scores"]
    assert score==expected


def test_resize_falls_back_without_changing_scores(media,model):
    kwargs=dict(options(),input_resolution="resize1_2",return_frame_scores=True)
    before=video.evaluate_video_gen(media/"ref.yuv",media/"dis.yuv",model,"cpu",inference_mode="legacy",**kwargs)
    after=video.evaluate_video_gen(media/"ref.yuv",media/"dis.yuv",model,"cpu",**kwargs)
    assert after==before and after[1]["inference_mode"]=="legacy"


@pytest.mark.parametrize("parallel",[False,True])
def test_full_official_metric_and_fifo_cleanup(media,model,tmp_path,parallel):
    kwargs=options();kwargs["dis_fmt"]="rgb"
    old=run_model(str(media/"ref.yuv"),str(media/"tagged.mkv"),model,"cpu",inference_mode="legacy",
                  n_threads=1,video_temp_path=str(tmp_path),**kwargs)
    new=run_model(str(media/"ref.yuv"),str(media/"tagged.mkv"),model,"cpu",n_threads=1,
                  parallel_branches=parallel,video_temp_path=str(tmp_path),**kwargs)
    assert old==new
    assert not list(tmp_path.rglob("*.yuv"))


def test_vmaf_conversion_is_a_pipe(media,tmp_path):
    with vmaf_inputs(media/"ref.yuv",media/"tagged.mkv","yuv420p","rgb",64,64,64,64,8,8,
                     temp_root=str(tmp_path)) as (paths,fmt,bits,processes):
        assert stat.S_ISFIFO(Path(paths[1]).stat().st_mode)
        assert Path(paths[1]).stat().st_size==0
        calc_vmaf(*paths,64,64,pix_fmt=fmt,bit_depth=bits,n_threads=1,
                  video_temp_path=str(tmp_path),producer_processes=processes)
    assert not list(tmp_path.rglob("*.yuv"))


def test_failed_conversion_does_not_hang_vmaf(media,tmp_path):
    broken=tmp_path/"broken.mp4";broken.write_bytes(b"invalid video")
    with pytest.raises(RuntimeError):
        with vmaf_inputs(media/"ref.yuv",broken,"yuv420p","rgb",64,64,64,64,8,8,
                         temp_root=str(tmp_path)) as (paths,fmt,bits,processes):
            calc_vmaf(*paths,64,64,pix_fmt=fmt,bit_depth=bits,n_threads=1,
                      video_temp_path=str(tmp_path),producer_processes=processes)
    assert not list(tmp_path.rglob("*.yuv"))


def test_benchmark_calls_production_api(monkeypatch):
    spec=importlib.util.spec_from_file_location("fdim_measurement",ROOT/"scripts/benchmark.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def inference(*args,**kwargs):
        calls.append(kwargs)
        return 2.,dict(raw_scores=[1.,1.],frame_indices=[0,2])
    monkeypatch.setattr(module,"evaluate_video_gen",inference)
    args=argparse.Namespace(ref="r",dis="d",ref_fmt="rgb",dis_fmt="rgb",width=64,height=64,
        interval=2,mode="stream",batch_size=1,prefetch=0,pipeline=False,uint8_transfer=True,
        frames=4,indices=[0,2],scope="deep")
    row=module.measure(args,object(),torch.device("cpu"))
    assert row["deep_score"]==2. and calls[0]["inference_mode"]=="stream"
    assert not hasattr(module,"stream_deep") and not hasattr(module,"StreamingReader")


def test_variable_frame_rate_sampling_uses_emitted_rgb_sequence(media,tmp_path):
    target=tmp_path/"vfr.mkv"
    subprocess.run(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","yuv420p",
        "-s","64x64","-r","2","-i",str(media/"dis.yuv"),"-vf",r"select=not(eq(n\,3))",
        "-vsync","0","-c:v","ffv1",str(target)],check=True)
    old=Reader(str(target),64,64,"rgb");frames=[]
    try:
        while True:
            ok,frame=old.next()
            if not ok:break
            frames.append(frame.copy())
    finally:old.close()
    new=SampledReader(target,64,64,"rgb",interval=2)
    try:
        for expected in frames[::2]:
            ok,actual=new.next()
            assert ok and np.array_equal(expected,actual)
        assert not new.next()[0]
    finally:new.close()


def test_10bit_keeps_legacy_deep_preprocessing(media,model,tmp_path):
    paths=[]
    for side in ['ref','dis']:
        target=tmp_path/(side+'.yuv')
        subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','yuv420p',
            '-s','64x64','-i',str(media/(side+'.yuv')),'-pix_fmt','yuv420p10le',
            '-f','rawvideo',str(target)],check=True)
        paths.append(target)
    kwargs=dict(options(),ref_fmt='yuv420p10le',dis_fmt='yuv420p10le',
                ref_bit_depth=10,dis_bit_depth=10,return_frame_scores=True)
    old=video.evaluate_video_gen(*paths,model,'cpu',inference_mode='legacy',**kwargs)
    new=video.evaluate_video_gen(*paths,model,'cpu',**kwargs)
    assert old==new and new[1]['inference_mode']=='legacy'


def test_hdr_default_retains_pu21_path(monkeypatch,model):
    class Source:
        closed=False
        def get_video_size(self):return 64,64,3
        def get_reference_frame(self,index,**kwargs):return torch.full((1,3,64,64),.5)
        def get_test_frame(self,index,**kwargs):return torch.full((1,3,64,64),.4)
        def close(self):self.closed=True
    sources=[]
    def create(*args,**kwargs):
        value=Source();sources.append(value);return value
    monkeypatch.setattr(video,'_build_pu21_video_source',create)
    old=video.evaluate_video_gen('ref.mp4','dis.mp4',model,'cpu',preprocess='pu21',
                                inference_mode='legacy',return_frame_scores=True)
    new=video.evaluate_video_gen('ref.mp4','dis.mp4',model,'cpu',preprocess='pu21',return_frame_scores=True)
    assert old==new and new[1]['inference_mode']=='legacy'
    assert all(source.closed for source in sources)
