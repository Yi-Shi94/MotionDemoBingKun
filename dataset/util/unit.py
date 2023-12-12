
def input_to_cm(x, in_unit):
    if in_unit in ['feet', 'foot']:
        x *= 30.48
    elif in_unit in ['m', 'meter']:
        x *= 100
    elif in_unit in ['cm', 'centermeter']:
        x *= 1.0
    else:
        x *= 1.0
        print('in_unit not implemented, scale as 1.0')
    return x

def cm_to_ouput(x, out_unit):
    # assume the input as cm, 'unit' as the target unit
    if out_unit in ['feet', 'foot']:
        scale = 1.0/30.48
    elif out_unit in ['m', 'meter']:
        scale = 1.0/100
    elif out_unit in ['cm', 'centermeter']:
        scale = 1.0
    else:
        scale = 1.0
        print('out_unit not implemented, scale as 1.0')
    return scale

def x_unit_transform(x, in_unit, out_unit):
    print('transform data from {} to {}',format(in_unit, out_unit))
    x_cm = input_to_cm(x, in_unit)
    x_out = cm_to_ouput(x, out_unit)
    return x_out

