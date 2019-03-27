
import tfmtcnn.tools.prepare_celeba_dataset as prep_celeba
import tfmtcnn.generate_simple_dataset as gen_simple
import tfmtcnn.generate_hard_dataset as gen_hard
import tfmtcnn.train_model as train_model
import mef


def main():
    prep_celeba_cmdline = f"--bounding_box_file_name ../data/CelebA/list_bbox_celeba.txt " \
        f"--landmark_file_name ../data/CelebA/list_landmarks_celeba.txt " \
        f"--output_file_name ../data/CelebA/CelebA.txt"

    mef.tsprint("Preparing CelebA dataset. Arguments:")
    mef.tsprint(prep_celeba_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = prep_celeba_cmdline.split(' ')
    # prep_celeba.main(prep_celeba.parse_arguments(argv))
    mef.tsprint("CelebA dataset preparation finished.")
    mef.tsprint("----------------------------------------------------")

    generate_simple_cmdline = f"--annotation_image_dir ../data/WIDER_Face/WIDER_train/images " \
        f"--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt " \
        f"--landmark_image_dir ../data/CelebA/images " \
        f"--landmark_file_name ../data/CelebA/CelebA.txt " \
        f"--base_number_of_images 700000 " \
        f"--target_root_dir ../data/datasets/mtcnn"

    mef.tsprint("Generating simple dataset for PNet training. Arguments:")
    mef.tsprint(generate_simple_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = generate_simple_cmdline.split(' ')
    gen_simple.main(gen_simple.parse_arguments(argv))
    mef.tsprint("PNet simple dataset generation finished.")
    mef.tsprint("----------------------------------------------------")

    train_pnet_cmdline = f"--network_name PNet " \
        f"--train_root_dir ../data/models/mtcnn/train " \
        f"--dataset_root_dir ../data/datasets/mtcnn " \
        f"--base_learning_rate 0.001 " \
        f"--max_number_of_epoch 19 " \
        f"--test_dataset FDDBDataset " \
        f"--test_annotation_image_dir ../../datasets/FDDB/ " \
        f"--test_annotation_file ../../datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt"

    mef.tsprint("Training PNet. Arguments:")
    mef.tsprint(train_pnet_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = train_pnet_cmdline.split(' ')
    train_model.main(train_model.parse_arguments(argv))
    mef.tsprint("PNet training finished.")
    mef.tsprint("----------------------------------------------------")

    generate_hard_rnet_cmdline = f"--network_name RNet " \
        f"--train_root_dir ../data/models/mtcnn/train " \
        f"--annotation_image_dir ../data/WIDER_Face/WIDER_train/images " \
        f"--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt " \
        f"--landmark_image_dir ../data/CelebA/images " \
        f"--landmark_file_name ../data/CelebA/CelebA.txt " \
        f"--base_number_of_images 700000 " \
        f"--target_root_dir ../data/datasets/mtcnn"

    mef.tsprint("Generating hard dataset for RNet training. Arguments:")
    mef.tsprint(generate_hard_rnet_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = generate_hard_rnet_cmdline.split(' ')
    gen_hard.main(gen_hard.parse_arguments(argv))
    mef.tsprint("RNet hard dataset generation finished.")
    mef.tsprint("----------------------------------------------------")

    train_rnet_cmdline = f"--network_name RNet " \
        f"--train_root_dir ../data/models/mtcnn/train " \
        f"--dataset_root_dir ../data/datasets/mtcnn " \
        f"--base_learning_rate 0.001 " \
        f"--batch_size 384 " \
        f"--max_number_of_epoch 22 " \
        f"--test_dataset FDDBDataset " \
        f"--test_annotation_image_dir ../../datasets/FDDB/ " \
        f"--test_annotation_file ../../datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt"

    mef.tsprint("Training RNet. Arguments:")
    mef.tsprint(train_rnet_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = train_rnet_cmdline.split(' ')
    train_model.main(train_model.parse_arguments(argv))
    mef.tsprint("RNet training finished.")
    mef.tsprint("----------------------------------------------------")

    generate_hard_onet_cmdline = f"--network_name ONet " \
        f"--train_root_dir ../data/models/mtcnn/train " \
        f"--annotation_image_dir ../data/WIDER_Face/WIDER_train/images " \
        f"--annotation_file_name ../data/WIDER_Face/WIDER_train/wider_face_train_bbx_gt.txt " \
        f"--landmark_image_dir ../data/CelebA/images " \
        f"--landmark_file_name ../data/CelebA/CelebA.txt " \
        f"--base_number_of_images 700000 " \
        f"--target_root_dir ../data/datasets/mtcnn"

    mef.tsprint("Generating hard dataset for ONet training. Arguments:")
    mef.tsprint(generate_hard_onet_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = generate_hard_onet_cmdline.split(' ')
    gen_hard.main(gen_hard.parse_arguments(argv))
    mef.tsprint("ONet hard dataset generation finished.")
    mef.tsprint("----------------------------------------------------")

    train_onet_cmdline = f"--network_name ONet " \
        f"--train_root_dir ../data/models/mtcnn/train " \
        f"--dataset_root_dir ../data/datasets/mtcnn " \
        f"--base_learning_rate 0.001 " \
        f"--max_number_of_epoch 21 " \
        f"--test_dataset FDDBDataset " \
        f"--test_annotation_image_dir ../../datasets/FDDB/ " \
        f"--test_annotation_file ../../datasets/FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt"

    mef.tsprint("Training ONet. Arguments:")
    mef.tsprint(train_onet_cmdline)
    mef.tsprint("----------------------------------------------------")
    argv = train_onet_cmdline.split(' ')
    train_model.main(train_model.parse_arguments(argv))
    mef.tsprint("ONet training finished.")
    mef.tsprint("----------------------------------------------------")
    return


if __name__ == '__main__':
    main()
