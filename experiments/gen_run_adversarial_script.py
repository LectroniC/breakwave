import sys
import os
import random
import math
import stat

#Forcing a random seed
random.seed(10)

# Example usages:
# python gen_adversarial_examples.py [input folder] [output folder] [restore model] [gen_script_loc] [batch_size] 
# 
# Script example:
# python adaptive_qin_attack.py --in sample-000000.wav --target "this is a test" --out adv_test.wav --restore_path DeepSpeech/deepspeech-0.4.1-checkpoint/model.v0.4.1
# wait
# python .... 

def main():
    if len(sys.argv) < 6:
        print("Number of arguments are wrong")
        exit()
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    restore_model_path = sys.argv[3]
    gen_script_loc = sys.argv[4]
    batch_size = int(sys.argv[5])

    wav_files = []
    orig_transcripts = []
    target_transcripts = []
    with open(os.path.join(input_folder, 'labels.csv'),'r') as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split(',')
            wav_file_name = splitted[0]
            target_transcript = splitted[-1]
            orig_transcript = splitted[-2]

            wav_files.append(wav_file_name)
            orig_transcripts.append(orig_transcript)
            target_transcripts.append(target_transcript)


    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    output_script_content = ""
    with open(gen_script_loc,'w') as file:
        content = ""
        for batch_i in range(math.ceil(len(wav_files)*1.0/batch_size)):
            content += "python simple_cw_attack "
            content += "--in "
            for j in range(batch_size):
                content += wav_files[batch_i*batch_size+j]
                content += " "
            content += "--target "
            for j in range(batch_size):
                content += '"'
                content += target_transcript[batch_i*batch_size+j]
                content += '"'
                content += " "
            content += "--out "
            for j in range(batch_size):
                content += "adv_"+wav_files[batch_i*batch_size+j]
                content += " "
            content += "--summary_csv "
            content += '"summary_{}.csv"'.format(batch_i)
            content += " "
            content += "--restore_path "
            content += restore_model_path
            content += "\n"
            content += "wait\n"

    with open(os.path.join(output_folder, 'labels.csv'),'w') as file:
        file.write(output_script_content)

    st = os.stat('somefile')
    os.chmod(os.path.join(output_folder, 'labels.csv'), st.st_mode | stat.S_IEXEC)    

main()