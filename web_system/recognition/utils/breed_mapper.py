import json
import os
from django.conf import settings

class BreedMapper:
    """
    狗品种映射工具类
    负责加载品种数据并提供中英文名称转换功能
    """

    _instance = None
    _breeds_data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._breeds_data is None:
            self._load_breeds_data()

    def _load_breeds_data(self):
        """
        加载品种数据文件
        """
        try:
            breeds_file = os.path.join(settings.BASE_DIR, 'static', 'data', 'dog_breeds.json')
            with open(breeds_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._breeds_data = data['breeds']
        except FileNotFoundError:
            raise FileNotFoundError("品种数据文件不存在: static/data/dog_breeds.json")
        except json.JSONDecodeError:
            raise ValueError("品种数据文件格式错误")

    def get_breed_list(self, format_type='en'):
        """
        获取所有品种列表

        Args:
            format_type: 'en' (英文), 'cn' (中文), 'both' (中英文组合)

        Returns:
            list: 品种名称列表
        """
        if format_type == 'en':
            return [breed['en'] for breed in self._breeds_data]
        elif format_type == 'cn':
            return [breed['cn'] for breed in self._breeds_data]
        elif format_type == 'both':
            return [f"{breed['cn']} ({breed['en']})" for breed in self._breeds_data]
        else:
            raise ValueError("format_type 必须是 'en', 'cn' 或 'both'")

    def get_breed_by_id(self, breed_id, format_type='en'):
        """
        根据ID获取品种名称

        Args:
            breed_id: 品种ID (0-75)
            format_type: 返回格式 'en', 'cn', 或 'dict'

        Returns:
            str or dict: 品种名称或完整信息
        """
        if not isinstance(breed_id, int) or not (0 <= breed_id < len(self._breeds_data)):
            raise ValueError(f"无效的品种ID: {breed_id}")

        breed = self._breeds_data[breed_id]

        if format_type == 'en':
            return breed['en']
        elif format_type == 'cn':
            return breed['cn']
        elif format_type == 'dict':
            return breed.copy()
        else:
            raise ValueError("format_type 必须是 'en', 'cn' 或 'dict'")

    def get_breed_by_name(self, name, name_type='en'):
        """
        根据名称获取品种信息

        Args:
            name: 品种名称
            name_type: 名称类型 'en' 或 'cn'

        Returns:
            dict: 品种完整信息，如果未找到返回None
        """
        for breed in self._breeds_data:
            if breed[name_type] == name:
                return breed.copy()
        return None

    def get_id_by_name(self, name, name_type='en'):
        """
        根据名称获取品种ID

        Args:
            name: 品种名称
            name_type: 名称类型 'en' 或 'cn'

        Returns:
            int: 品种ID，如果未找到返回-1
        """
        breed_info = self.get_breed_by_name(name, name_type)
        return breed_info['id'] if breed_info else -1

    def get_display_name(self, breed_name, name_type='en'):
        """
        获取品种的显示名称（中英文组合）

        Args:
            breed_name: 品种英文或中文名称
            name_type: 输入名称类型 'en' 或 'cn'

        Returns:
            str: "中文名 (英文名)" 格式
        """
        breed_info = self.get_breed_by_name(breed_name, name_type)
        if breed_info:
            return f"{breed_info['cn']} ({breed_info['en']})"
        return breed_name  # 如果找不到，返回原名称

    def get_total_breeds(self):
        """
        获取品种总数

        Returns:
            int: 品种数量
        """
        return len(self._breeds_data)

# 全局单例实例
breed_mapper = BreedMapper()