import json
import pytest
from physicalview.paper_require_complete import require_complete
from physicalview.paper_capture import FEATURES


def test_dataset_gate_rejects_incomplete_images_even_if_feature_says_captured(tmp_path):
    rows=[{'scene':f'scene{i}'} for i in range(10)]
    (tmp_path/'roster.json').write_text(json.dumps({'datasets':{'scannetpp':rows}}))
    for row in rows:
        folder=tmp_path/'scannetpp'/row['scene'];folder.mkdir(parents=True)
        features={}
        for key in FEATURES:
            image=folder/key/'frame.png';image.parent.mkdir();image.write_bytes(b'fixture')
            image.with_suffix('.json').write_text('{}')
            features[key]={'status':'captured','frames':[f'{key}/frame.png']}
        (folder/'capture-complete.json').write_text(json.dumps({'features':features}))
    result=require_complete(tmp_path,'scannetpp')
    assert result['captured_groups']==140
    assert result['publication_quality_verified'] is False
    (tmp_path/'scannetpp/scene9/l_robot/frame.json').unlink()
    with pytest.raises(RuntimeError,match='scene9/l_robot:missing'):
        require_complete(tmp_path,'scannetpp')


def test_bounded_retry_archives_replaced_group_and_retains_unrelated_evidence(tmp_path):
    from physicalview.paper_capture import prepare_feature_retry
    records={'a_original':{'status':'captured'},'h_prompt_inpaint':{'status':'captured'}}
    (tmp_path/'features.json').write_text(json.dumps(records))
    for key in records:
        folder=tmp_path/key;folder.mkdir();(folder/'frame.png').write_bytes(b'original pixels')
        (folder/'review.json').write_text('{"status":"approved"}')
    archive=prepare_feature_retry(tmp_path,['h_prompt_inpaint'])
    assert (archive/'h_prompt_inpaint/frame.png').read_bytes()==b'original pixels'
    assert not (tmp_path/'h_prompt_inpaint').exists()
    assert (tmp_path/'a_original/frame.png').read_bytes()==b'original pixels'
    assert json.loads((tmp_path/'features.json').read_text())==records
