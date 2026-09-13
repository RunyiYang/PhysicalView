"""Build a review gallery and editable SVG panels without modifying scene pixels."""
import argparse,base64,csv,html,json
from pathlib import Path
from PIL import Image
from physicalview.paper_capture import FEATURES
from physicalview.phiview import save_json


def panel(folder,feature,paths,sidecar):
    # Image data remains lossless; text/caption is a separate editable vector layer.
    picks=paths if len(paths)<=3 else [paths[0],paths[len(paths)//2],paths[-1]]
    cell_w=960;cell_h=640;gap=24;top=70;bottom=75
    width=cell_w*len(picks)+gap*(len(picks)-1)
    parameter=feature=='e_physical_parameters'
    if parameter:width+=520
    doc=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{top+cell_h+bottom}" viewBox="0 0 {width} {top+cell_h+bottom}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="12" y="42" font-family="DejaVu Sans,sans-serif" font-size="28">{html.escape(FEATURES[feature])}</text>']
    for i,p in enumerate(picks):
        x=i*(cell_w+gap);im=Image.open(p);w,h=im.size;dh=min(cell_h,cell_w*h/w);dw=dh*w/h
        encoded=base64.b64encode(p.read_bytes()).decode()
        doc += [f'<image x="{x}" y="{top}" width="{dw}" height="{dh}" href="data:image/png;base64,{encoded}"/>',
                f'<text x="{x+12}" y="{top+cell_h+32}" font-family="DejaVu Sans,sans-serif" font-size="24">{html.escape(p.stem)}</text>']
    if parameter:
        data=json.loads((folder/'parameters.json').read_text());x=cell_w+32
        lines=[('Selected object',sidecar['state']['selected']),('Mass (kg)',f"{data['mass_kg']:.4g}"),('Sliding friction',f"{data['friction'][0]:.4g}"),('Inertia (kg m²)',', '.join(f'{v:.3g}' for v in data['inertia_kg_m2'])),('Source','Active MuJoCo model')]
        for i,(label,value) in enumerate(lines):
            y=top+45+i*105
            doc += [f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="20" fill="#64748b">{html.escape(label)}</text>',f'<text x="{x}" y="{y+35}" font-family="DejaVu Sans,sans-serif" font-size="25">{html.escape(str(value))}</text>']
    doc.append('</svg>');(folder/'panel.svg').write_text('\n'.join(doc))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root).resolve();rows=[];cards=[]
    for dataset in ['scannetpp','libero','behavior']:
        for scene in sorted((root/dataset).glob('*')):
            if not scene.is_dir():continue
            file=scene/'features.json';features=json.loads(file.read_text()) if file.exists() else {}
            for key,title in FEATURES.items():
                row=features.get(key,{'status':'not_run','frames':[]})
                paths=[scene/p for p in row.get('frames',[]) if (scene/p).exists()]
                review_path=scene/key/'review.json'
                review=json.loads(review_path.read_text()) if review_path.exists() else {'status':'pending'}
                rows.append({'dataset':dataset,'scene':scene.name,'feature':key,'status':row['status'],'pngs':len(paths),'review':review['status'],'error':row.get('error','')})
                if paths:
                    side=json.loads(paths[0].with_suffix('.json').read_text());panel(scene/key,key,paths,side)
                    link=(scene/key/'panel.svg').relative_to(root)
                    cards.append(f'<article data-dataset="{dataset}" data-feature="{key}"><h2>{dataset} / {scene.name} / {key}</h2><p>{html.escape(title)} · {row["status"]} · publication review: {html.escape(review["status"])}</p><a href="{link}"><img loading="lazy" src="{link}"></a><p><a href="{scene.relative_to(root)}/features.json">Evidence</a> · <a href="{scene.relative_to(root)}/{key}/">Lossless PNG and sidecars</a></p></article>')
    save_json(root/'coverage.json',rows)
    with (root/'coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','scene','feature','status','pngs','review','error']);w.writeheader();w.writerows(rows)
    doc='''<!doctype html><meta charset="utf-8"><title>PhiView paper figure review</title><style>body{font-family:system-ui;margin:40px;background:#f4f6f9;color:#152238}header{position:sticky;top:0;background:white;padding:16px;z-index:2}article{background:white;padding:20px;margin:24px 0;border-radius:10px}h2{font-size:18px}img{width:100%;height:auto}select{padding:8px;margin:8px}p{color:#475569}</style><header><h1>PhiView — paper figure review</h1><p>Actual renders and simulation states. Captured does not mean approved for publication. See evidence for source resolution and model/GT provenance.</p><select id="dataset"><option value="">All datasets</option><option>scannetpp</option><option>libero</option><option>behavior</option></select><select id="feature"><option value="">All features</option>'''
    doc+=''.join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in FEATURES.items())+'</select></header>'+''.join(cards)
    doc+='''<script>function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=(dataset.value&&a.dataset.dataset!==dataset.value)||(feature.value&&a.dataset.feature!==feature.value))}dataset.onchange=feature.onchange=filter;</script>'''
    (root/'gallery.html').write_text(doc)
    print(json.dumps({'scene_feature_rows':len(rows),'captured':sum(r['status']=='captured' for r in rows),'review_approved':sum(r['review']=='approved' for r in rows)}))

if __name__=='__main__':main()
