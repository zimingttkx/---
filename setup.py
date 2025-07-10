"""
这个文件用于设置Python包的安装配置。
它定义了包的名称、版本、作者、描述等信息，并指定了需要安装的依赖包。
包括项目的元数据和依赖项,配置了包的安装方式和依赖关系。

"""
# 这个没有什么用 是我用来测试git的
# find_packages()用于自动查找当前目录下的所有包和子包。
# install_requires用于指定安装该包时需要的依赖项。
# setup()函数用于定义包的元数据和安装配置。
# 例如，name指定包的名称，version指定版本号，author指定作者等
from setuptools import setup,find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    这个函数用于读取requirements.txt文件中的依赖项，并返回一个包含所有依赖项的列表。
    :param file_path: requirements.txt文件的路径
    :return:
    """
    requirement_list:list[str] = []
    try:
        with open("requirements.txt","r") as file:
            # 读取文件内容
            lines = file.readlines()
            # 去除每行的空格和换行符
            for line in lines:
                requirement = line.strip()
                # 如果行不为空且不是注释，则添加到依赖列表中
                if requirement and requirement!= '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        raise Exception(f"{requirement.txt} not found")

    return requirement_list

setup(
    name="Network Security Project Based on Python and ML", # 包的名称
    version="0.0.1", # 包的版本
    author="梓铭", # 包的作者
    author_email= "2147514473@qq.com",
    packages = find_packages(), # 包含的包
    install_requires = get_requirements() # 依赖项
)