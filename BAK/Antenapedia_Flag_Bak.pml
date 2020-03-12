load BAK_model1.pdb

select pep_antenapedia, resid 1-16
select pep_flag, resid 17-24
select pep_bak, resid 25-46

#set_color c_pep_antenapedia, [25, 114, 120]
set_color c_pep_antenapedia, [0, 136, 204]
set_color c_pep_flag, [226, 47, 47]
set_color c_pep_bak, [112, 183, 126]

bg_color white

hide everything
set cartoon_color, c_pep_antenapedia, pep_antenapedia
set cartoon_color, c_pep_flag, pep_flag
set cartoon_color, c_pep_bak, pep_bak
show cartoon, all

alter 38-47/, ss='H'
alter 30-35/, ss='H'


# get_view (in pymol)
set_view (\
     0.537566841,    0.549137771,    0.639876544,\
    -0.190158278,    0.818253696,   -0.542457879,\
    -0.821486712,    0.169932693,    0.544301987,\
     0.000389898,   -0.000146737,  -86.120422363,\
    -4.676537991,    3.862828493,   20.134841919,\
    76.256607056,   95.968383789,  -20.000000000 )

set depth_cue, 0
set antialias, 2

set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
#ray 1200,900
ray 2400,1800
png Antenapedia_flag_bak.png, dpi=300

