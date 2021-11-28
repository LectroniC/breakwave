from posixpath import split
import sys
import os
import shutil
import random

#Forcing a random seed
random.seed(10)

#Example usages:
#python draw_from_commonvoice_subset.py [input_path] [output_path] [number of samples extracted] 

def main():
    if len(sys.argv) < 4:
        print("Number of arguments are wrong")
        exit()
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    number_of_samples = int(sys.argv[3])

    wav_files = [f for f in os.listdir(input_folder) if '.wav' in f and os.path.isfile(os.path.join(input_folder, f))]
    wav_files_sampled = random.sample(wav_files, number_of_samples)
    target_transcript = []
    with open(os.path.join(input_folder, 'labels.csv'),'r') as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split(',')
            wav_file_name = splitted[0]
            transcript = splitted[-1]
            if wav_file_name not in wav_files_sampled:
                target_transcript.append(transcript)
    
    target_transcript = random.sample(target_transcript, number_of_samples)


    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    output_label_csv_content = ""

    with open(os.path.join(input_folder, 'labels.csv'),'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            splitted = lines[i].split(',')
            if splitted[0] in wav_files_sampled:
                output_label_csv_content += lines[i].strip()
                output_label_csv_content += ","
                output_label_csv_content += target_transcript[i]
                output_label_csv_content += "\n"
                shutil.copy(os.path.join(input_folder, splitted[0]), 
                            os.path.join(output_folder, splitted[0]))

    with open(os.path.join(output_folder, 'labels.csv'),'w') as file:
        file.write(output_label_csv_content)

main()