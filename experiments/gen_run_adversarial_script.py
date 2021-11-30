import sys
import os
import random
import math
import stat

#Forcing a random seed
random.seed(10)

# Example usages:
# python gen_adversarial_examples.py [use_script_name] [input folder] [new labels csv] [restore model] [gen_script_name] [batch_size] 
# 
# Script example:
# python adaptive_qin_attack.py --in sample-000000.wav --target "this is a test" --out adv_test.wav --restore_path DeepSpeech/deepspeech-0.4.1-checkpoint/model.v0.4.1
# wait
# python .... 

def main():
    if len(sys.argv) < 7:
        print("Number of arguments are wrong")
        exit()
    
    use_script_name = sys.argv[1]
    input_folder = sys.argv[2]
    new_labels_csv = sys.argv[3]
    restore_model_path = sys.argv[4]
    gen_script_name = sys.argv[5]
    batch_size = int(sys.argv[6])

    wav_files = []
    orig_transcripts = []
    target_transcripts = []
    new_target_transcripts = []
    with open(os.path.join(input_folder, 'labels.csv'),'r') as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split(',')
            print(splitted)
            wav_file_name = splitted[0].strip()
            target_transcript = splitted[-1].strip()
            orig_transcript = splitted[-2].strip()

            wav_files.append(wav_file_name)
            orig_transcripts.append(orig_transcript)
            target_transcripts.append(target_transcript)

    with open(gen_script_name,'w') as file:
        content = "#!/bin/sh\n"
        for batch_i in range(math.ceil(len(wav_files)*1.0/batch_size)):
            content += "python {} ".format(use_script_name)
            content += "--in "
            curr_batch_size = min(len(wav_files)-batch_i*batch_size, batch_size)
            for j in range(curr_batch_size):
                content += os.path.abspath(os.path.join(input_folder, wav_files[batch_i*batch_size+j]))
                content += " "
            content += "--target "
            
            # We only take one target for a whole batch.
            content += '"'
            content += target_transcripts[batch_i*batch_size]
            for _ in range(curr_batch_size):
                new_target_transcripts.append(target_transcripts[batch_i*batch_size])
            content += '"'
            content += " "
            
            content += "--out "
            for j in range(curr_batch_size):
                content += "adv_"+wav_files[batch_i*batch_size+j]
                content += " "
            content += "--summary_csv "
            content += 'summary_{}.csv'.format(batch_i)
            content += " "
            content += "--restore_path "
            content += restore_model_path
            content += "\n"
            content += "wait\n"
        file.write(content)
    
    st = os.stat(gen_script_name)
    os.chmod(gen_script_name, st.st_mode | stat.S_IEXEC)    


    with open(new_labels_csv,'w') as file:
        content = ""
        for i in range(len(lines)):
            splitted = lines[i].split(',')
            splitted = [i.strip() for i in splitted]
            splitted[-1] = new_target_transcripts[i]
            content += ",".join(splitted)
            content += "\n"
        file.write(content)

main()