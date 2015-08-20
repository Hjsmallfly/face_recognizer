# face_recognizer
face_recognizer.py 

1)创建csv文件
read_csv(filename, images, labels, labelsInfo, sep=';')
文件格式为 <path>;<label>[;<comment>]\n  用于创建人脸识别器时提供图片的标签以及图片文件的路径

2)训练分类器
train(csv_file, filenameTosave=TRAIN_RESUT)
读取csv文件中的 路径 以及 标签，最终训练的结果存储于 filenameTosave 这个文件中。

3)预测
predict(filenames, train_result, print_out=False)
返回包含元素为 (文件名，标签，准确度) 的 list 列表

predict_dir(root='.', pattern="*.jpg", train_result=TRAIN_RESUT)
用于预测指定文件夹中指定类型图片
