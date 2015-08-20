# face_recognizer

[face_recognizer.py]
1)创建csv文件
read_csv(filename, images, labels, labelsInfo, sep=';')
文件格式为，用于创建人脸识别器时提供图片的标签以及图片文件的路径
<path>;<label>[;<comment>]\n  

2)训练分类器
train(csv_file, filenameTosave=TRAIN_RESUT)
读取csv文件中的 路径 以及 标签，最终训练的结果存储于 filenameTosave 这个文件中。

3)预测
predict(filenames, train_result, print_out=False)
返回包含元素为 (文件名，标签，准确度) 的 list 列表

predict_dir(root='.', pattern="*.jpg", train_result=TRAIN_RESUT)
用于预测指定文件夹中指定类型图片

[face_process.py]
1)判断传入的图片(cv2读入的图片格式)中是否存在人脸，返回各个人脸的坐标
any_faces(image)
判断标准是，能成功找到眼睛

2)在图片中裁出人脸，判断过程在其中已经包含
cropFaces(image, size=(100, 100), grayscale=False)

3)对指定目录里面的图片进行人脸提取
process_dir(dirpath=".", pattern="*.jpg", folder_for_each_picture=True, delete_src=False)
folder_for_each_picture  # 决定是否将每张图片中的人脸都保存在单独的一个文件夹中
delete_src  # 决定在裁剪好之后是否删除原来的输入图像

4)通过摄像头捕捉照片
record_face(amount=-1, dirname=None)
拍摄amount张图片(内部调用了 any_faces())，若检测到人脸则自动存储照片到dirname文件夹中

[face_interaction.py]

对上述两个模块进行简单的应用。
    demo = FaceInteraction(10, 'nwad')  # 10 people at most, 'nwad' is the name of the folder the program uses
    demo.predict_on_video()  # or demo.train()
建立名字为nwad的文件夹，里面保存了所有train留下的数据

    
