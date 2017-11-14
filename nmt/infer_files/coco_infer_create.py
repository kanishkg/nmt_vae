
def multiple_paraphrases(filename):
    """
    Creates a test file with multiple occurrences of the same sentence
    """
    fid = open(filename, "rb")
    mult_pp = []
    for line in fid.readlines():
        for i in range(4):
            mult_pp.append(line)

    fid.close()
    return mult_pp


def store_in_file(arr, filename):
    fid = open(filename, "w")
    for sent in arr:
        fid.write(sent)
    fid.close()

mpp = multiple_paraphrases("./coco_test.src")
# mcv = multiple_controls("./coco_ct_test.src")
store_in_file(mpp, "./coco_infer_mpp.src")
mpp = multiple_paraphrases("./coco_test.tgt")
store_in_file(mpp, "./coco_infer_mpp.tgt")

# store_in_file(mcv, "./coco_ct_infer_mcv.src")
