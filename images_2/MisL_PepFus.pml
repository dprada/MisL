load MisL_PepFus.pdb

remove solvent

select protein, chain A
select pep_fus, res 1-29 and chain A
select pep_flag, res 30-39 and chain A
select pep_bax, res 40-61 and chain A
select pep_ompt, res 62-67 and chain A

select membrane_up, resname DUM and name O
select membrane_down, resname DUM and name N

set_color c_protein, [255,165,0]
set_color c_pep_fus, [5, 11, 181]
set_color c_pep_flag, [226, 47, 47]
set_color c_pep_bax, [30, 142, 0]
set_color c_pep_ompt, [28, 28, 28]
set_color c_gray_memb, [200, 200, 200]

bg_color white

hide everything
show cartoon, protein
set cartoon_color, c_protein, protein
set cartoon_color, c_pep_fus, pep_fus
set cartoon_color, c_pep_flag, pep_flag
set cartoon_color, c_pep_bax, pep_bax
set cartoon_color, c_pep_ompt, pep_ompt

#show spheres, membrane_up
#show spheres, membrane_down
#set sphere_scale, 0.3
#set sphere_color, c_gray_memb, membrane_up
#set sphere_color, c_gray_memb, membrane_down

alter 30-39/, ss='H'
rebuild

# get_view (in pymol)

set_view (\
     0.954210103,    0.013603600,   -0.298712075,\
     0.298716992,   -0.088585034,    0.950199366,\
    -0.013529832,   -0.995930612,   -0.088596940,\
    -0.001429061,   -0.003529698, -410.114227295,\
    17.252464294,    1.263007641,  -38.088878632,\
   310.462097168,  509.768615723,  -20.000000000 )

set depth_cue, 0
set antialias, 2

set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
ray 900,900
png MisL_PepFus.png, dpi=300

