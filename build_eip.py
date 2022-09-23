local = True

addmod_submod_low=None
addmod_submod_all=None
mulmont_low=None
mulmont_all=None
setmod_all=None

if local:
    md_image_tag="!{{AddMod SubMod}}[/{}]"
    addmod_submod_low=md_image_tag.format('charts/addmod_submod_low.png')
    addmod_submod_all=md_image_tag.format('charts/addmod_submod_all.png')
    mulmont_low=md_image_tag.format('charts/mulmont_low.png')
    mulmont_all=md_image_tag.format('charts/mulmont_all.png')
    setmod_all=md_image_tag.format('charts/setmod_all.png')

with open('eip.md.template') as f:
    template = ''.join(f.readlines())
    template = template.format(addmod_submod_low=addmod_submod_low, addmod_submod_all=addmod_submod_all, mulmont_low=mulmont_low, mulmont_all=mulmont_all, setmod_all=setmod_all)
    print(template)
