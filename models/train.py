# 微调模型的函数
def train_and_save_model(x_train, y_train, n_steps, Size, n_filters, n_kernel_size, model_path, fine_tune=False, base_model_path=None):
    if fine_tune and base_model_path:
        # 加载模型时指定自定义对象
        model = load_model(base_model_path, custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention, 'r_squared': r_squared})
        # 如果是微调，加载预训练模型
        for layer in model.layers[:5]:  # 冻结前N层
            layer.trainable = False
        # 微调时通常降低学习率
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[r_squared])
    else:
        model = create_model(n_steps, Size, n_filters, n_kernel_size)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[r_squared])
        # 打印模型总结
        model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=60, batch_size=64, callbacks=[early_stopping, model_checkpoint, reduce_lr])

    return model