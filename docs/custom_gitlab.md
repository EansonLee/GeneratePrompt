# 自定义GitLab实例支持指南

本文档介绍如何配置环境变量以支持自定义GitLab实例，包括企业级GitLab和私有GitLab服务器。

## 概述

系统支持通过API访问GitLab仓库，无需完整克隆仓库。除了标准的GitLab.com外，还支持自定义GitLab实例，如企业私有GitLab服务器。

## 配置步骤

### 1. 添加自定义GitLab域名

有两种方式可以添加自定义GitLab域名：

#### 方式一：环境变量

在`.env`文件中添加以下配置：

```
# 自定义GitLab域名（多个域名用逗号分隔）
CUSTOM_GITLAB_DOMAINS=git.example.com,gitlab.company.org
```

#### 方式二：配置文件

系统会自动识别以下常见GitLab域名特征：
- 包含"gitlab"的域名 
- 包含"git"的域名

如需添加更多自定义域名，可以修改`config/config.py`中的`GIT_API_CONFIG`配置。

### 2. 配置访问令牌

对于私有仓库，需要配置访问令牌。可以通过以下两种方式配置：

#### 通用令牌

对所有GitLab实例使用同一个令牌：

```
# 通用GitLab API令牌
GITLAB_API_TOKEN=your_gitlab_token
```

#### 特定域名令牌

为特定GitLab实例配置专用令牌（推荐）：

```
# 特定域名的令牌
GITLAB_API_TOKEN_GIT_EXAMPLE_COM=your_token_for_example
GITLAB_API_TOKEN_GITLAB_COMPANY_ORG=your_token_for_company
```

注意：环境变量名是将域名中的`.`和`-`替换为`_`，并全部转为大写，然后加上`GITLAB_API_TOKEN_`前缀。

## 使用多级路径结构

对于使用多级路径结构的GitLab实例，系统已经自动处理。例如，如下路径结构都可以正确识别：

- `http://git.example.com/owner/repo`
- `http://git.example.com/group/subgroup/project`

## 常见问题

### 无法访问仓库

如果遇到404错误，可能有以下原因：

1. **路径不正确**：确认仓库URL是否正确
2. **权限不足**：确认访问令牌是否有足够权限
3. **分支不存在**：系统会自动尝试常见分支名（main, master, develop, dev）

### 安全注意事项

1. 不要在公共环境泄露你的访问令牌
2. 推荐使用只读访问令牌，只授予最小必要权限
3. 定期轮换访问令牌

## 验证配置

配置完成后，系统启动时会输出检测到的自定义GitLab域名和加载的令牌信息，可以通过日志验证配置是否生效。

## 示例

### 示例1：企业GitLab

```
CUSTOM_GITLAB_DOMAINS=git.company.com
GITLAB_API_TOKEN_GIT_COMPANY_COM=glpat-xxxxxxxxxxxxxxxx
```

### 示例2：自托管GitLab

```
CUSTOM_GITLAB_DOMAINS=gitlab.internal.network
GITLAB_API_TOKEN_GITLAB_INTERNAL_NETWORK=glpat-xxxxxxxxxxxxxxxx
``` 