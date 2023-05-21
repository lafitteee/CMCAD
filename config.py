import torch


class Config:
    def __init__(self):

        # 数据文件参数
        self.command_file_path = './dataset/command_2_60.npy'
        self.path_file_path = './dataset/path_norm_2_60.npy'
        self.all_data_path = './dataset/dataset_with_image_norm.npz'
        self.clf_data_dir = './dataset/data_classification.npz'
        self.sim_data_dir = './dataset/data_id_cmd_arg.npz'
        self.images_dir = './dataset/images'
        self.save_path = './save_model/'
        self.saved_model_path = './save_model/DeepCAD_2023-02-03-18-41-23_acc_0.82.pkl'  # 预测模式加载模型的路径
        self.teacher_model_path = './save_model\教师模型_交叉注意力融合_2023-02-02-01-20-41_acc_0.81.pkl'  # CMCAD模型保存路径
        self.img_encoder_model_path = './save_model\学生模型_基于DeepSVG_2023-03-01-02-03-57_acc_0.25.pkl'  # 图像编码模型的加载路径
        self.decoder_output_path = './output/'           # decoder输出保存目录
        self.topK_path = 'save_model/cosine_similarity_top10_0624_WithPosEmbed.pkl'  # 预测模式 topK 保存路径


        # 数据参数
        self.command = ['START', 'LINE', 'ARC', 'CIRCLE', 'END']
        self.max_path = 62
        self.arg_num = 7
        self.arg_dim = 256  # 每一个参数数值
        self.command_type_num = len(self.command)

        # command参数
        self.command2slice = {
            1: [slice(0, 4)],               # LINE
            2: [slice(0, 2), slice(4, 7)],  # ARC
            3: [slice(0, 2), slice(6, 7)],   # CIRCLE
        }
        self.command2argnum = {
            1: 4,
            2: 5,
            3: 3
        }
        self.CMD_ARGS_MASK = torch.tensor([[0, 0, 0, 0, 0, 0, 0],   # <START>
                                           [1, 1, 1, 1, 0, 0, 0],    # LINE
                                           [1, 1, 0, 0, 1, 1, 1],    # ARC
                                           [1, 1, 0, 0, 0, 0, 1],   # CIRCLE
                                           [0, 0, 0, 0, 0, 0, 0]]   # <END>
                                          )
                                              

        # 训练参数
        self.batch_size = 32
        self.epoch = 200
        self.learning_rate = 0.0005
        self.image_encoder_lr = 0.001
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.save_per_epoch = 200
        self.dropout = 0.1

        # 模型参数
        self.n_layer = 4
        self.d_model = 256
        self.nhead = 8

        self.use_vae = False
        # self.mode = "eval"
        # self.mode = "train"
        # self.mode = "output_decoder"
        # self.mode = "image_trainer"
        # self.mode = "train_CMCAD"
        # self.mode = "eval_CMCAD"
        # self.mode = 'train_img_encoder_base_CMCAD'
        # self.mode = 'eval_img_encoder_base_CMCAD'
        self.mode = 'calculate_similarity'

