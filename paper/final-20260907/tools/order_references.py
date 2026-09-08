"""Order explicit LaTeX bibitems by first citation; reject missing/unused keys."""
import argparse
from pathlib import Path
import re


def order(path):
    text = path.read_text(encoding='utf8')
    start = re.search(r'\\begin\{thebibliography\}\{\d+\}', text)
    end = text.index(r'\end{thebibliography}', start.end())
    body = text[:start.start()]
    keys = []
    for match in re.finditer(r'\\cite\{([^}]+)\}', body):
        for key in match[1].split(','):
            key = key.strip()
            if key not in keys:
                keys.append(key)
    bib = text[start.end():end]
    matches = list(re.finditer(r'\\bibitem\{([^}]+)\}', bib))
    items = {m[1]:bib[m.start():matches[i+1].start() if i+1<len(matches) else len(bib)].strip()
             for i,m in enumerate(matches)}
    if len(items) != len(matches) or set(items) != set(keys):
        raise ValueError('duplicate, missing or unused bibliographic key')
    ordered = (body + r'\begin{thebibliography}{'+str(len(keys))+'}\n'+
               bib[:matches[0].start()].strip()+'\n'+
               '\n'.join(items[key] for key in keys)+'\n'+text[end:])
    path.write_text(ordered, encoding='utf8', newline='\n')
    print('References in first-citation order: '+', '.join(keys))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    order(parser.parse_args().source)
