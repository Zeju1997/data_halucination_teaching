# 导入第三方库
import argparse
import yaml
import os
 
 
# 创建对象并初始化参数
def get_parser():
    
    # 创建解析对象
    parser = argparse.ArgumentParser(description = 'A method to update parser parameters using yaml files') 
    
    # 添加参数1
    parser.add_argument(
        '--num_worker', 
        type = int,
        default = 4,
        help = 'the number of worker for data loader')
    
    # 添加参数2
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01,
        help = 'the learning rate of SGD')
    
    # 添加参数3
    parser.add_argument(
    '--batchsize',
    type = int,
    default = 64,
    help = 'the batchsize of training stage')
    
    # 返回解析对象
    return parser 
 
 
 
# 创建一个yaml文件
def creat_yaml():
    
    # yaml文件存放的内容
    caps = {
        'num_worker': 16,
        'lr': 0.05,
    }
 
    # yaml文件存放的路径
    yamlpath = os.path.join('./', 'test.yaml')
 
    # caps的内容写入yaml文件
    with open(yamlpath, "w", encoding = "utf-8") as f:
        yaml.dump(caps, f)
 
        
# main()函数    
def main():
    
    # 创建一个yaml文件
    creat_yaml()
    
    # 创建解析对象
    parser = get_parser()
    
    # 实例化对象
    p = parser.parse_args(args = [])
    
    # 输出参数原始默认值
    print('The default value of num_worker is: ', p.num_worker)
    print('The default value of lr is: ', p.lr)
    print('The default value of batchsize is: ', p.batchsize)
    print('##############################')
    
    # 导入创建的yaml文件
    with open('test.yaml', 'r') as f:
        default_arg = yaml.load(f)
        
    # 创建解析对象
    parser = get_parser() 
    
    # 利用yaml文件更新默认值
    parser.set_defaults(**default_arg)
    
    # 实例化对象
    p = parser.parse_args(args = [])
 
    # 输出更新后的参数值
    print('The updated value of num_worker is: ', p.num_worker)
    print('The updated value of lr is: ', p.lr)
    print('The updated value of batchsize is: ', p.batchsize) # batchsize并没有用yaml文件更新哦
 
 
# 执行main函数
main()
