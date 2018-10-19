#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable    
__kernel void minp( __global uint4 *src,                                   

                  __global uint  *gmin,                                 

                  __local uint *lmin,                                

                  __global uint  *dbg,                                  

                  uint    nitems,                               

                  uint    dev )                                

{                                                                   

uint  count  = ( nitems / 4 ) / get_global_size(0);                 

uint  idx   =	get_global_id(0);                           

uint  stride =  get_global_size(0);                     

uint  pmin   = (uint) -1;                                              
                                                                         
for( int n=0; n < count; n++, idx += stride )                          

  {                                                                   

     pmin = min( pmin, src[idx].x );                                  

     pmin = min( pmin, src[idx].y );                                     

     pmin = min( pmin, src[idx].z );                                    

     pmin = min( pmin, src[idx].w );                                  

  }                                                                    

if( get_local_id(0) == 0 )                                               

     lmin[0] = (uint) -1;                                                

                                                                         

  barrier( CLK_LOCAL_MEM_FENCE );                                    
                                                                   

   (void) atom_min( lmin, pmin );                                     

                                                                     
  barrier( CLK_LOCAL_MEM_FENCE );                                        

if( get_local_id(0) == 0 )                                           

      gmin[ get_group_id(0) ] = lmin[0];                                

if( get_global_id(0) == 0 )                                         

  {                                                                      

    dbg[0] = get_num_groups(0);                                      

	dbg[1] = get_global_size(0);                                      

    dbg[2] = count;                                                

    dbg[3] = stride;                                              

  }                                                                    
}                                                                       

__kernel void reduce( __global uint4 *src,                               

                     __global uint  *gmin )                             

{                                                                        

   (void) atom_min( gmin, gmin[get_global_id(0)] ) ;                     

}                                                                       
