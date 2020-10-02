import os
import json

log_dir = 'saved_models'
model_dir = 'GAN_pg23_pd10_g1_d1_1.0'
result_file = 'results.json'
result_file = os.path.join(log_dir, model_dir, result_file)
results = json.load(open(result_file))
results = results['results']

output_file = 'results.txt'
output_file = os.path.join(log_dir, model_dir, output_file)

fout = open(output_file, 'w')
for i in range(0, len(results), 400):
    fout.write('%4d.\tI:  %6d\n' % (i, results[i]['image_id']))
    fout.write('Q: %s  /  A_pred: %s\n' % (results[i]['question'], results[i]['answer']))
    fout.write('E_gt  : %s\n' % results[i]['explain_gt'])
    fout.write('E_pred: %s\n\n' % results[i]['explain_res'])

fout.close()
