import re
import os

# 读取文件
try:
    with open('src/agents/design_prompt_agent.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复_initialize_llm方法中的缩进问题
    pattern1 = r'try:\s+return ChatOpenAI\('
    repl1 = 'try:\n                return ChatOpenAI('
    content = re.sub(pattern1, repl1, content)
    
    # 修复__init__方法中的缩进问题
    pattern2 = r'self.vector_store = vector_store'
    repl2 = '            self.vector_store = vector_store'
    content = re.sub(pattern2, repl2, content)
    
    # 修复其他缩进问题
    pattern3 = r'self._load_prompt_cache\(\)'
    repl3 = '            self._load_prompt_cache()'
    content = re.sub(pattern3, repl3, content)
    
    # 修复工作流初始化的缩进
    pattern4 = r'self.workflow = None'
    repl4 = '            self.workflow = None'
    content = re.sub(pattern4, repl4, content)
    
    # 修复except块的缩进问题
    pattern5 = r'except Exception as e:\s+logger.error'
    repl5 = 'except Exception as e:\n            logger.error'
    content = re.sub(pattern5, repl5, content)
    
    # 修复_retrieve_history_prompts方法中的缩进问题
    pattern6 = r'return state\s+except Exception as e:'
    repl6 = 'return state\n        except Exception as e:'
    content = re.sub(pattern6, repl6, content)
    
    # 修复_extract_tech_stack_components方法中的缩进问题
    pattern7 = r'return state\s+except Exception as e:'
    repl7 = 'return state\n        except Exception as e:'
    content = re.sub(pattern7, repl7, content)
    
    # 修复_evaluate_prompt方法中的缩进问题
    pattern8 = r'return state\s+except Exception as e:'
    repl8 = 'return state\n        except Exception as e:'
    content = re.sub(pattern8, repl8, content)
    
    # 保存修复后的文件
    with open('src/agents/design_prompt_agent.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print('文件修复成功！')
except Exception as e:
    print(f'修复过程中出错: {str(e)}')
