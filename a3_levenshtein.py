import os
import numpy as np
import re

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    N = len(r)
    M = len(h)

    # special cases
    if N == 0 and M == 0:
        return (float(0), 0, 0, 0)
    if N == 0 and M != 0:
        return (np.inf, 0, M, 0)
    if N != 0 and M == 0:
        return (1.0, 0, 0, N)

    # Initialize R and B
    R = np.zeros((N+1, M+1))
    B = np.zeros((N+1, M+1), dtype=object) 

    R[0, :] = np.arange(0, M + 1)
    R[:, 0] = np.arange(0, N + 1) 
    B[0, :] = 'left' 
    B[:, 0] = 'up' 
    B[0,0] = ''

    # update B and R based on # of errors and type of errors
    for i in range(1, N+1):
        for j in range(1, M+1):
            del_error = R[i-1, j] + 1

            sub_error = R[i-1, j-1]
            if r[i-1] != h[j-1]:
                sub_error += 1

            ins_error = R[i, j-1] + 1

            # find the minimum sum
            R[i, j] = min(del_error, sub_error, ins_error)

            # update B
            if R[i, j] == del_error:
                B[i, j] = 'up'
            elif R[i, j] == ins_error:
                B[i, j] = 'left'
            else:
                B[i, j] = 'up-left'

    # count number of deletions, insertions and substitutions based on the updated B
    ins_counts = 0
    del_counts = 0
    row = N
    col = M

    while row > 0 or col > 0:
        val = B[row, col]
        if val == 'up':
            del_counts += 1
            row -= 1
        if val == 'left':
            ins_counts += 1
            col -= 1
        if val == 'up-left':
            row -= 1
            col -= 1

    # calculate WER and number of substitutions
    WER = R[N, M] / float(N)
    sub_counts = int(np.rint(R[N, M] - ins_counts - del_counts))

    return (WER, sub_counts, ins_counts, del_counts)

# helper function for removing unnecessary punctuations. 
def preprocess(line):

    # dealing with the case of empty string
    if len(line.split(" ", 2)) != 3:
        return ''
    # ignore [i] and [LABEL]
    line = line.split(" ", 2)[2]

    # remove all punctuations except []
    line = re.sub(r"[!\"#\$%&\'()*\+,-\.\/:;<=>?@\^_`{|}~\\]+", " ", line)

    # remove extra brackets 
    line = re.sub(r"(\[)+", "[", line)
    line = re.sub(r"(\])+", "]", line)

    # remove the puncuation [ (ie. no ] at the end of the token)
    line = re.sub(r"(\[)(?![\w]+\])", "", line)

    # remove the puncuation ] (ie. no [ at the beginning of the token)
    def removeRightBrackets(match):
        if match.group(1)[0] != "[":
            return match.group(1)
        else:
            return match.group(0)
    pattern = re.compile(r"([\S]+)(\]+)")
    line = re.sub(pattern, removeRightBrackets, line)

    line = line.strip()
    return line

# return empty list if and only if the transcipt is ['']
def remove_empty(transcripts):
    if len(transcripts) == 1 and transcripts[0] == '':
        return []
    else:
        return transcripts

if __name__ == "__main__":
    google_errors = []
    kaldi_errors = []
    with open("asrDiscussion.txt", "a+") as output_f:
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print(speaker)
                speaker_path = os.path.join(dataDir, speaker)
                trans_path = os.path.join(speaker_path, 'transcripts.txt')
                google_path = os.path.join(speaker_path, 'transcripts.Google.txt')
                kaldi_path = os.path.join(speaker_path, 'transcripts.Kaldi.txt')
                # read files and remove empty transcript ie. will skip the script if it's empty
                trans = open(trans_path, 'r')
                transcripts = remove_empty(trans.read().strip().lower().split('\n'))
                trans.close()

                google_trans = open(google_path, 'r')
                google_lines = remove_empty(google_trans.read().strip().lower().split('\n'))
                google_trans.close()

                kaldi_trans = open(kaldi_path, 'r')
                kaldi_lines = remove_empty(kaldi_trans.read().strip().lower().split('\n'))
                kaldi_trans.close()

                # use the minimum length
                min_len = min(len(transcripts), len(google_lines), len(kaldi_lines))

                # compute error rate for each lines of both google transcript and kaldi transcript
                for i in range(0, min_len):
                    # preprocess line to remove unnecessary punctuations
                    ref = preprocess(transcripts[i]).split()
                    google = preprocess(google_lines[i]).split()
                    kaldi = preprocess(kaldi_lines[i]).split()
                    
                    # apply levenshtein to get WERs
                    google_result = Levenshtein(ref, google)
                    kaldi_result = Levenshtein(ref, kaldi)
                    google_errors.append(google_result[0])
                    kaldi_errors.append(kaldi_result[0])

                    # save to file
                    output_f.write('{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6} \n'.format(speaker, 'Google', i, 
                        google_result[0], google_result[1], google_result[2], google_result[3]))

                    output_f.write('{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6} \n'.format(speaker, 'Kaldi', i, 
                        kaldi_result[0], kaldi_result[1], kaldi_result[2], kaldi_result[3]))
                output_f.write('\n')

        # comput std and mean for both Kaldi and Google, and save to file
        output_f.write('Kaldi WER Average: {0: 1.4f}, Kaldi WER Standard Deviation: {1: 1.4f} \n'.format(
            np.mean(kaldi_errors), np.std(kaldi_errors)))
        output_f.write('Google WER Average: {0: 1.4f}, Google WER Standard Deviation: {1: 1.4f} \n'.format(
            np.mean(google_errors), np.std(google_errors)))
        # write discussions here to avoid overwrite
        output_f.write('By looking at the mean and standard deviation we got for Kaldi and Google, we can tell that Kaldi has smaller mean and standard deviation. \n')
        output_f.write('In this case, Kaldi seems to perform better than Google on our data as it has smaller mean, and the kaldi is less variable because of the smaller standard deviation. \n')
        output_f.write("By manually examining the transcript files, I found that Google always tries to produce a more complete and readable sentence by ignoring non-word terms, which means many entries like 'umm', <LG> or <BR> in the original transcripts will be ignored in the Google transcripts, causing the high deletion errors and substitution errores. \n")
        output_f.write('And Kaldi has terms like [laughter] or [noise], sometimes those terms are used to replace the terms likes <BR> <LG> in the reference, which related to the substitution error. \n')
        output_f.close()
