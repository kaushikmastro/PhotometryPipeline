KPL/MK

\begintext

Dawn mission metakernel template.

This file is intended to be rendered by replacing {{KERNEL_DIRECTORY}} with the
absolute or project-relative directory that contains the downloaded SPICE
kernels. After templating, the rendered file should be a valid NAIF metakernel.

Kernel inventory categories covered here:
- LSK: leap seconds
- PCK: planetary constants
- SPK: ephemerides
- CK: attitude
- DSK: shape models

\begindata

PATH_SYMBOLS = ( 'KERNEL_DIRECTORY' )
PATH_VALUES  = ( '{{KERNEL_DIRECTORY}}' )

KERNELS_TO_LOAD = (

   /* LSK */
   '$KERNEL_DIRECTORY/naif0012.tls'

   /* PCK */
   '$KERNEL_DIRECTORY/pck00010.tpc'

   /* SPK */
   '$KERNEL_DIRECTORY/de421.bsp'
   '$KERNEL_DIRECTORY/sb_vesta_nav_120628.bsp'

   /* CK */
   '$KERNEL_DIRECTORY/dawn_sc_071008_071014_v2.bc'
   '$KERNEL_DIRECTORY/dawn_sa_110103_110109.bc'
   '$KERNEL_DIRECTORY/dawn_vir_072901530_0.bc'

   /* DSK */
   '$KERNEL_DIRECTORY/vesta_shape.dsk'
)

\begintext

Notes:
- The CK and DSK filenames above are template placeholders for the Dawn mission
  kernel set. Replace them with the locally available mission kernels you want
  to load if your kernel cache uses different names.
- Keep this template tracked in Git; rendered metakernels should remain ignored.
