"""Keep one consistent BEHAVIOR object state and preserve native RGB resolution.

Camera registration and state filtering use dataset-provided mesh trajectories;
this is GT-assisted pose evidence, even though object discovery uses AUTO masks.
"""
import argparse,json
from pathlib import Path
import numpy as np


def select_consistent_clips(registered, pose_to_matrix, translation_tol=.02, rotation_tol_degrees=5.):
    episodes={}
    for clip in registered:episodes.setdefault(clip['ep'],[]).append(clip)
    trials=[]
    for episode,clips in sorted(episodes.items()):
        clips=sorted(clips,key=lambda c:c['clip_id']);reference=clips[0]
        poses={k:reference['D']@pose_to_matrix(np.asarray(t)[0]) for k,t in reference['traj'].items()}
        keep=[];audit=[]
        for clip in clips:
            shared=set(poses)&set(clip['traj']);distances=[];angles=[]
            for key in shared:
                current=clip['D']@pose_to_matrix(np.asarray(clip['traj'][key])[0]);base=poses[key]
                distances.append(float(np.linalg.norm(current[:3,3]-base[:3,3])))
                angles.append(float(np.rad2deg(np.arccos(np.clip((np.trace(current[:3,:3]@base[:3,:3].T)-1)/2,-1,1)))))
            delta=max(distances,default=float('inf'));angle=max(angles,default=float('inf'))
            accepted=len(shared)>=3 and delta<=translation_tol and angle<=rotation_tol_degrees
            if accepted:keep.append(clip)
            audit.append({'clip':clip['clip_id'],'shared_meshes':len(shared),'max_translation_m':delta if distances else None,'max_rotation_degrees':angle if angles else None,'kept':accepted})
        trials.append({'episode':episode,'reference':reference['clip_id'],'kept':keep,'audit':audit})
    if not trials:raise ValueError('No registered BEHAVIOR clips')
    best=max(trials,key=lambda t:sum(len(c['cam_frames']) for c in t['kept']))
    report={'selected_episode':best['episode'],'reference_clip':best['reference'],
        'registered_clips':len(registered),'selected_clips':len(best['kept']),
        'selected_camera_frames':sum(len(c['cam_frames']) for c in best['kept']),
        'translation_tolerance_m':translation_tol,'rotation_tolerance_degrees':rotation_tol_degrees,
        'selection':'first static state of one episode, maximizing consistent observed views before reconstruction',
        'pose_source':'dataset mesh trajectories, GT-assisted',
        'episodes':[{k:v for k,v in t.items() if k!='kept'} for t in trials]}
    return best['kept'],report


def main():
    from oracle import capture_generator as source
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--scene-name',required=True);ap.add_argument('--root',required=True);ap.add_argument('--episodes',type=int,default=6);ap.add_argument('--max-frames',type=int,default=120);ap.add_argument('--tasks-config',required=True);a=ap.parse_args()
    scene=Path(a.root)/'data'/a.scene_name;scene.mkdir(parents=True,exist_ok=True)
    original_register=source._register_clips_to_reference
    def register(clips):
        registered=original_register(clips);kept,report=select_consistent_clips(registered,source.pose7_to_mat)
        (scene/'pose-selection.json').write_text(json.dumps(report,indent=2))
        if report['selected_camera_frames']<8:
            raise ValueError('Fewer than eight posed frames of one consistent scene state; do not fuse different object states to manufacture coverage')
        return kept
    source._register_clips_to_reference=register
    source.extract_task(a.task,a.scene_name,a.root,episodes=a.episodes,static_only=True,max_frames=a.max_frames,tasks_config=a.tasks_config,upscale=1)
    calibration=json.loads((scene/'dslr/nerfstudio/transforms_undistorted.json').read_text())
    (scene/'paper-source-resolution.json').write_text(json.dumps({'width':calibration['w'],'height':calibration['h'],'upscale':1,'original_encoded_rgb_bytes_preserved':True,'source':'PointWorld-BEHAVIOR WDS initial RGB frames'},indent=2))

if __name__=='__main__':main()
