"""枚举类型模块"""
from enum import Enum
from typing import List

class DesignType(str, Enum):
    """设计类型枚举"""
    UI = "UI"  # 用户界面
    UX = "UX"  # 用户体验
    MOBILE = "Mobile"  # 移动应用
    WEB = "Web"  # 网页
    DASHBOARD = "Dashboard"  # 仪表盘
    FORM = "Form"  # 表单
    LANDING = "Landing"  # 着陆页
    OTHER = "Other"  # 其他

class TechStack(str, Enum):
    """技术栈枚举"""
    ANDROID = "Android"  # 安卓
    IOS = "iOS"  # 苹果iOS
    FLUTTER = "Flutter"  # Flutter跨平台
    REACT = "React"  # React
    VUE = "Vue"  # Vue.js
    ANGULAR = "Angular"  # Angular
    REACT_NATIVE = "React Native"  # React Native
    SWIFT = "Swift"  # Swift
    KOTLIN = "Kotlin"  # Kotlin
    OTHER = "Other"  # 其他 