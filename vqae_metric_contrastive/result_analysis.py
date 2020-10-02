import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', type=str, default='saved_models/VQAE_VQA_baseline')
    parser.add_argument('--B', type=str, default='saved_models/VQAE_(vq*e\\vq*e)_masked_unk_num')
    parser.add_argument('--A_name', type=str, default='Base')
    parser.add_argument('--B_name', type=str, default='Mask')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    res_A = json.load(open(os.path.join(args.A, 'results.json')))['results']
    res_B = json.load(open(os.path.join(args.B, 'results.json')))['results']
    assert len(res_A) == len(res_B)

    res_A = sorted(res_A, key=lambda x:x['question_id'])
    res_B = sorted(res_B, key=lambda x:x['question_id'])

    neq = []
    ArBr = []
    for i in range(len(res_A)):
        assert(res_A[i]['question_id'] == res_B[i]['question_id'])
        if res_A[i]['answer_pred'] != res_B[i]['answer_pred']:
            neq.append({'index': i,
                        'gt_ans': res_A[i]['answer_gt'],
                        'A_ans': res_A[i]['answer_pred'],
                        'B_ans': res_B[i]['answer_pred'],
                        'question': res_A[i]['question'],
                        'image_id': res_A[i]['image_id'],
                        'explain_gt': res_B[i]['explain_gt'],
                        'question_id': res_A[i]['question_id']})
        elif res_A[i]['answer_gt'] == res_A[i]['answer_pred']:
            ArBr.append({'index': i,
                        'gt_ans': res_A[i]['answer_gt'],
                        'A_ans': res_A[i]['answer_pred'],
                        'B_ans': res_B[i]['answer_pred'],
                        'question': res_A[i]['question'],
                        'image_id': res_A[i]['image_id'],
                        'explain_gt': res_B[i]['explain_gt'],
                        'question_id': res_A[i]['question_id']})

    ArBw = []
    AwBr = []
    for n in neq:
        if n['gt_ans'] == n['A_ans'] and n['gt_ans'] != n['B_ans']:
            ArBw.append(n)
        elif n['gt_ans'] != n['A_ans'] and n['gt_ans'] == n['B_ans']:
            AwBr.append(n)

    ArBw_path = os.path.join('results', args.A_name + '_right_' + args.B_name + '_wrong.json')
    AwBr_path = os.path.join('results', args.A_name + '_wrong_' + args.B_name + '_right.json')
    ArBr_path = os.path.join('results', args.A_name + args.B_name + '_right.json')
    json.dump(ArBw, open(ArBw_path, 'w'))
    json.dump(AwBr, open(AwBr_path, 'w'))
    json.dump(ArBr, open(ArBr_path, 'w'))

    ArBw_cnt = []
    ArBw_color = []
    ArBw_yesno = []
    ArBw_where = []
    for n in ArBw:
        if 'how many' in n['question'].lower():
            ArBw_cnt.append(n)
        elif 'where is' in n['question'].lower():
            ArBw_where.append(n)
        elif 'what color is' in n['question'].lower():
            ArBw_color.append(n)
        elif n['gt_ans'] == 'yes' or n['gt_ans'] == 'no':
            ArBw_yesno.append(n)

    ArBw_cnt_path = os.path.join('results', args.A_name[0]+'r'+args.B_name[0]+'w'+'_cnt.json')
    ArBw_where_path = os.path.join('results', args.A_name[0]+'r'+args.B_name[0]+'w'+'_where.json')
    ArBw_color_path = os.path.join('results', args.A_name[0]+'r'+args.B_name[0]+'w'+'_color.json')
    ArBw_yesno_path = os.path.join('results', args.A_name[0]+'r'+args.B_name[0]+'w'+'_yesno.json')
    json.dump(ArBw_cnt, open(ArBw_cnt_path, 'w'))
    json.dump(ArBw_color, open(ArBw_color_path, 'w'))
    json.dump(ArBw_yesno, open(ArBw_yesno_path, 'w'))
    json.dump(ArBw_where, open(ArBw_where_path, 'w'))

    AwBr_cnt = []
    AwBr_color = []
    AwBr_yesno = []
    AwBr_where = []
    for n in AwBr:
        if 'how many' in n['question'].lower():
            AwBr_cnt.append(n)
        elif 'where is' in n['question'].lower():
            AwBr_where.append(n)
        elif 'what color is' in n['question'].lower():
            AwBr_color.append(n)
        elif n['gt_ans'] == 'yes' or n['gt_ans'] == 'no':
            AwBr_yesno.append(n)

    AwBr_cnt_path = os.path.join('results', args.A_name[0]+'w'+args.B_name[0]+'r'+'_cnt.json')
    AwBr_where_path = os.path.join('results', args.A_name[0]+'w'+args.B_name[0]+'r'+'_where.json')
    AwBr_color_path = os.path.join('results', args.A_name[0]+'w'+args.B_name[0]+'r'+'_color.json')
    AwBr_yesno_path = os.path.join('results', args.A_name[0]+'w'+args.B_name[0]+'r'+'_yesno.json')
    json.dump(AwBr_cnt, open(AwBr_cnt_path, 'w'))
    json.dump(AwBr_color, open(AwBr_color_path, 'w'))
    json.dump(AwBr_yesno, open(AwBr_yesno_path, 'w'))
    json.dump(AwBr_where, open(AwBr_where_path, 'w'))
