from dataloaders.train_validation_test_loader import TrainValTestLoader


def dataloader_factory(mode, dataset, dataset_input_path, num_classes, output_path,
                       model_name, reference_crop_size, reference_stride_crop,
                       simulate_dataset, mean=None, std=None):
    if dataset == 'road_detection':
        return TrainValTestLoader(mode, dataset, dataset_input_path, num_classes, output_path,
                                  model_name, reference_crop_size, reference_stride_crop,
                                  simulate_dataset, mean, std)
    else:
        raise NotImplementedError('DataLoader not identified: ' + dataset)
