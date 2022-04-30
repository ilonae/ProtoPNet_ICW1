base_architecture = 'resnet18'
img_size = 224
prototype_shape = (70, 512, 1, 1)
num_classes = 7
train_batch_size = 32
test_batch_size = 30
train_push_batch_size = 32

prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

data_path = './datasets/cq500/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'


joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 10
num_warm_epochs = 1

push_start = 2
push_epochs = [i for i in range(num_train_epochs) if i % 2 == 0]
