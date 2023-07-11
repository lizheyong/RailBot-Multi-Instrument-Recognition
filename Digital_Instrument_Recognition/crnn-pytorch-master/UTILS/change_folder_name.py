import os

path = r'C:\Users\zheyong\Desktop\digitaaaaaa'
# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)

for i in fileList:
    dushu, b = i.split('_')
    num = b.split('.')[0]  # num是数字顺序
    new_num = str( int(num) + 2205 )
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + i  # os.sep添加系统分隔符
    print('sss')
    # 设置新文件名
    newname = path + os.sep + dushu + '_' + new_num + '.jpg'

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)