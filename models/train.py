def train_and_save_model(x_train, y_train, n_steps, Size, n_filters, n_kernel_size, model_path, fine_tune=False, base_model_path=None):
    if fine_tune and base_model_path:
        model = load_model(base_model_path, custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention, 'r_squared': r_squared})
        for layer in model.layers[:5]:  
            layer.trainable = False
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[r_squared])
    else:
        model = create_model(n_steps, Size, n_filters, n_kernel_size)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[r_squared])
        model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=60, batch_size=64, callbacks=[early_stopping, model_checkpoint, reduce_lr])

    return model
