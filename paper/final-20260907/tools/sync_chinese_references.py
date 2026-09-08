"""Use the English bibliography as the Chinese companion's sole reference source."""
from pathlib import Path
import re


def plain(value):
    value=value.replace(r'\allowbreak', '')
    value=value.replace(r'{\"o}', 'ö').replace('``', '“').replace("''", '”')
    value=value.replace(r'\&','&').replace(r'\%', '%').replace('~',' ').replace('--','–')
    value=re.sub(r'\\url\{([^}]+)\}', lambda m:'['+m[1]+']('+m[1]+')', value)
    value=re.sub(r'\\(?:emph|texttt)\{([^}]+)\}', r'\1',value)
    value=value.replace(r'\bibitem', '')
    if '\\' in value:
        raise ValueError('unhandled bibliography TeX: '+value)
    return re.sub(r'\s+',' ',value).strip()


def main():
    root=Path(__file__).resolve().parents[1]
    source=(root/'english/main.tex').read_text(encoding='utf8')
    bib=source.split(r'\begin{thebibliography}',1)[1].split(r'\end{thebibliography}',1)[0]
    entries=re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)',bib,re.S)
    target=root/'chinese/中文解读.md'
    text=target.read_text(encoding='utf8').split('\n## 参考文献',1)[0].rstrip()
    text+='\n\n## 参考文献\n\n以下条目与英文稿按首次引用次序一致；保留英文题名便于查找原文。\n\n'
    text+='\n\n'.join('- ['+str(i)+'] '+plain(content) for i,(key,content) in enumerate(entries,1))+'\n'
    target.write_text(text,encoding='utf8',newline='\n')
    print('Chinese bibliography synchronized: '+str(len(entries))+' references')


if __name__=='__main__':
    main()
