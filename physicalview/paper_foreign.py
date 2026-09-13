"""Dataset adapters and real construction stages before the common capture runner."""
import argparse,json,os
from pathlib import Path
from physicalview.config import load_config
from physicalview import pipeline
from physicalview.paper_campaign import run_command,stage
from physicalview.phiview import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--index',type=int,required=True);ap.add_argument('--dataset',choices=['libero','behavior'],required=True)
    a=ap.parse_args();root=Path(a.root).resolve();cfg=load_config(root/f'{a.dataset}-config.yaml')
    row=json.loads((root/f'{a.dataset}-roster.json').read_text())[a.index];sid=row['scene'];out=root/a.dataset/sid;logs=out/'construction-logs';logs.mkdir(parents=True,exist_ok=True)
    scene=cfg.scannetpp_root/'data'/sid;asset=cfg.outputs_root/(sid+'_factory');asset.mkdir(parents=True,exist_ok=True)
    ctx=pipeline.StageContext(cfg,sid,asset,auto=a.dataset=='behavior',scene_dir=scene)
    if a.dataset=='libero':
        if not (scene/'native-capture.json').exists():
            run_command([cfg.interpreter('studio'),'-m','physicalview.paper_libero','--roster',root/'libero-roster.json','--index',a.index,'--root',cfg.scannetpp_root],{},cfg.package_root,logs,'native-capture')
    else:
        if not (scene/'dslr/colmap/images.txt').exists():
            run_command([cfg.interpreter('main'),'-m','oracle.capture_generator','extract','--task',row['task'],'--scene-name',sid,'--root',cfg.scannetpp_root,'--episodes',6,'--max-frames',120,'--tasks-config',root/'behavior-tasks.yaml'],{},cfg.repo_root,logs,'wds-extract')
    splat=cfg.splats_root/(sid+'.ply')
    if not splat.exists():
        run_command([cfg.interpreter('gsplat'),'-m','agents.recon.gsplat_train','--scene-dir',scene,'--init-ply',scene/'init_points.ply','--out',splat,'--iters',15000],{},cfg.repo_root,logs,'gaussian-train')
    if a.dataset=='behavior':
        env=ctx.base_env();env['SIMANY_MESH_SRC']='derived'
        if not (asset/'derived_mesh.ply').exists():
            for step,key in [('render','gsplat'),('fuse','main')]:
                run_command([cfg.interpreter(key),'-m','agents.discover.derive_mesh_from_splat',step],env,cfg.repo_root,logs,'derived-'+step)
        discovery=pipeline.discover(ctx,'sam3_auto')
        for spec in discovery:
            spec.env['SIMANY_MESH_SRC']='derived'
            if 'agents.discover.auto_segment' in spec.argv:spec.argv += ['--mesh-path',str(asset/'derived_mesh.ply'),'--frame-stride','1']
    else:discovery=pipeline.discover(ctx,'gt_segments')
    if not (asset/'objects/objects.json').exists():
        for i,spec in enumerate(discovery):stage(spec,logs,f'discover-{i}')
    objects=json.loads((asset/'objects/objects.json').read_text())
    if not objects:raise ValueError('No objects survived discovery; no replacement scene fabricated')
    if not all((asset/'objects'/f"obj_{r['index']:02d}"/'trellis_gs.ply').exists() for r in objects):
        spec=pipeline.generate(ctx,'trellis',None)
        if a.dataset=='behavior':spec.env['SIMANY_MESH_SRC']='derived'
        stage(spec,logs,'trellis-generation')
    if not (asset/'objects/aligned_all.json').exists():
        spec=pipeline.register(ctx,'yaw_sweep_icp',None)
        if a.dataset=='behavior':spec.env['SIMANY_MESH_SRC']='derived'
        stage(spec,logs,'registration')
    metadata=[]
    for obj in objects:
        n=f"obj_{obj['index']:02d}";p=asset/'objects'/n
        alignment=json.loads((p/'aligned.json').read_text())
        metadata.append({'id':n,'label':obj['label'],'accepted':not bool(alignment.get('rejected')),'physics':(p/'physics.json').exists()})
    rows=root/f'{a.dataset}-rows';rows.mkdir(exist_ok=True)
    save_json(rows/f'{a.index}.json',{'dataset':a.dataset,'scene':sid,'result_set':sid+'_factory','assets':asset,'objects':metadata,'auto':a.dataset=='behavior','source':row})
    env={'SIMANY_MESH_SRC':'derived'} if a.dataset=='behavior' else {}
    run_command([cfg.interpreter('studio'),'-m','physicalview.paper_campaign','--root',root,'--index',a.index,'--dataset',a.dataset],env,cfg.package_root,logs,'feature-campaign')

if __name__=='__main__':main()
