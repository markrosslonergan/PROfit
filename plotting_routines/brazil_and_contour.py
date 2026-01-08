import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import uproot
from scipy.ndimage import binary_closing
import random

add_external = True

#fig_name = 'spine_external.pdf'
#fig_name = 'external_only.pdf'
fig_name = 'sensitivity.pdf'
#fig_name = 'pandora_external.pdf'
#fig_name = 'pandora_external_only.pdf'
#fig_name = 'spine_data.pdf'
#fig_name = 'pandora_data.pdf'
#fig_name = 'spine_brazil.pdf'
#fig_name = 'pandora_brazil.pdf'

#data_label = 'SPINE Data'
#data_label = 'Pandora Data'
#data_label = 'ICARUS (Pandora) 90%'
data_label = 'ICARUS (SPINE) 90%'

fc_file = 'feldman_surf_60x60.root'
#fc_file = 'fine_feldman_surf.root'
#gfc_name = 'pandora_90'
gfc_name = 'spine_90'

data_fc_file = 'fine_feldman_surf_fixed.root'
#data_fc_name = 'pandora_90'
data_fc_name = 'spine_90'

#universes_file = 'pandora_universes.pdf'
universes_file = 'spine_universes.pdf'

#contour_file = "pandora_brazil_4k_surf.root"
#data_file = "pandora_run2_surf.root"
#asimov_file = "pandora_asimov_surf.root"
contour_file = "spine_brazil_4k_surf.root"
data_file = "spine_run2_surf.root"
asimov_file = "spine_asimov_surf.root"

ext_data = {'MiniBooNE' : 'numu_MiniBoone_Sciboone_2012.csv',
            'CCFR' : 'numu_CCFR_2012.csv',
            'CDGS' : 'numu_CDHS_2012.csv',
            'IceCube' : 'icecube_95_closed.csv',
            'NOvA' : 'nova_2024_mumu.csv',
            'MINOS' : ('MINOS_data.root','minosPlus90CL_0')} 

def make_data_contour(root_file):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
    ax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

    with uproot.open(data_fc_file) as fcfile:
        fcgrid = fcfile[data_fc_name].values()
        #fcgrid = fcgrid.transpose()
    
    with uproot.open(root_file) as file:
        hist = file['surf']
        values = hist.values()
        #values = values - np.min(values)
        values = values - fcgrid

        x_edges = hist.axis(0).edges()
        y_edges = hist.axis(1).edges()
        
        # Calculate bin centers (log space)
        x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
        y_centers = np.sqrt(y_edges[:-1] * y_edges[1:])
        
        pts = (values > -0.6) * (values < 0.6) 
        for i in range(len(x_edges)-1):
            for j in range(len(y_edges)-1):
                if pts[i,j]:
                    print(f'{i} {j}')

        interp = RegularGridInterpolator(
            (np.log10(x_centers), np.log10(y_centers)),
            #values - np.nanmin(values),
            values,
            method='cubic',  
            bounds_error=False,
            fill_value=np.nan  
        )
         
        # Create grid
        N = 100
        x_min, x_max = np.log10(x_edges[0]*1.05), np.log10(x_edges[-1]*0.95)
        y_min, y_max = np.log10(y_edges[0]*1.05), np.log10(y_edges[-1]*0.95)
        xg = np.linspace(x_min, x_max, N)
        yg = np.linspace(y_min, y_max, N)
        xg_mesh, yg_mesh = np.meshgrid(xg, yg)

        # Interpolate
        zg = interp((xg_mesh, yg_mesh))
        zg = np.ma.masked_invalid(zg)
        
        # Plot and get contours
        #cs = ax.contour(10**xg, 10**yg, zg, levels=[4.61], colors=[color], linewidths=2)
        cs = ax.contour(10**xg, 10**yg, zg, levels=[0], colors=['black'], linewidths=2)
            
        contour_paths = []
        for contour in cs.allsegs[0]:  
            if len(contour) > 1:  # Only keep meaningful contours
                contour_paths.append(contour)
        return contour_paths

    #ax.set_xlim(0.05, 1)
    #ax.set_ylim(1e-2, 100)
    #plt.tight_layout()
    #plt.show()
    #
    #return all_contours

def make_fakedata_contour(root_file):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
    ax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

    with uproot.open(fc_file) as fcfile:
        fcgrid = fcfile[fc_name].values()
        fcgrid = fcgrid.transpose()
    
    with uproot.open(root_file) as file:
        hist = file['brazil_throw_surf_111;1']
        values = hist.values()
        values = values - np.min(values)
        values = values - fcgrid

        x_edges = hist.axis(0).edges()
        y_edges = hist.axis(1).edges()
        
        # Calculate bin centers (log space)
        x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
        y_centers = np.sqrt(y_edges[:-1] * y_edges[1:])
        
        pts = (values > -0.6) * (values < 0.6) 
        for i in range(len(x_edges)-1):
            for j in range(len(y_edges)-1):
                if pts[i,j]:
                    print(f'{i} {j}')

        interp = RegularGridInterpolator(
            (np.log10(x_centers), np.log10(y_centers)),
            #values - np.nanmin(values),
            values,
            method='cubic',  
            bounds_error=False,
            fill_value=np.nan  
        )
         
        # Create grid
        N = 100
        x_min, x_max = np.log10(x_edges[0]*1.05), np.log10(x_edges[-1]*0.95)
        y_min, y_max = np.log10(y_edges[0]*1.05), np.log10(y_edges[-1]*0.95)
        xg = np.linspace(x_min, x_max, N)
        yg = np.linspace(y_min, y_max, N)
        xg_mesh, yg_mesh = np.meshgrid(xg, yg)

        # Interpolate
        zg = interp((xg_mesh, yg_mesh))
        zg = np.ma.masked_invalid(zg)
        
        # Plot and get contours
        #cs = ax.contour(10**xg, 10**yg, zg, levels=[4.61], colors=[color], linewidths=2)
        cs = ax.contour(10**xg, 10**yg, zg, levels=[0], colors=['black'], linewidths=2)
            
        contour_paths = []
        for contour in cs.allsegs[0]:  
            if len(contour) > 1:  # Only keep meaningful contours
                contour_paths.append(contour)
        return contour_paths


def plot_and_save_contours(root_file, fc_name=gfc_name, hist_pattern="surf"):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
    ax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

    all_contours = {}

    pts = None 
    count = 0

    pts_x_edges = None
    pts_y_edges = None

    with uproot.open(fc_file) as fcfile:
        fcgrid = fcfile[fc_name].values()
        fcgrid = fcgrid.transpose()
    
    pp = PdfPages(universes_file)
    with uproot.open(root_file) as file:
        #hist_keys = [key for key in file.keys() if key.startswith(hist_pattern)]
        hist_keys = [key for key in file.keys() if key.startswith('brazil_throw')]
        colors = plt.cm.viridis(np.linspace(0, 1, len(hist_keys)))

        for hist_key, color in zip(hist_keys, colors):
        #for hist_key in hist_keys:
            hist = file[hist_key]
            values = hist.values()
            values = values - np.min(values)
            values = values - fcgrid
            #values = values - 4.61

            if pts is None:
                pts = np.zeros(values.shape, dtype=float)

            if values[0,0] < 0:
                count += 1
                pts += values >= 0

#            print(f'{hist_key} \n\n {values}')
            x_edges = hist.axis(0).edges()
            y_edges = hist.axis(1).edges()
            if pts_x_edges is None: pts_x_edges = x_edges
            if pts_y_edges is None: pts_y_edges = y_edges
            
            # Calculate bin centers (log space)
            x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
            y_centers = np.sqrt(y_edges[:-1] * y_edges[1:])
            
            interp = RegularGridInterpolator(
                (np.log10(x_centers), np.log10(y_centers)),
                #values - np.nanmin(values),
                values,
                method='cubic',  
                bounds_error=False,
                fill_value=np.nan  
            )
             
            # Create grid
            N = 100
            x_min, x_max = np.log10(x_edges[0]*1.05), np.log10(x_edges[-1]*0.95)
            y_min, y_max = np.log10(y_edges[0]*1.05), np.log10(y_edges[-1]*0.95)
            xg = np.linspace(x_min, x_max, N)
            yg = np.linspace(y_min, y_max, N)
            xg_mesh, yg_mesh = np.meshgrid(xg, yg)

            # Interpolate
            zg = interp((xg_mesh, yg_mesh))
            zg = np.ma.masked_invalid(zg)
            
            #if values[0,0] < 0: continue
            #tfig, tax = plt.subplots(figsize=(7, 7), dpi=150)
            #tax.set(xscale='log', yscale='log')
            #tax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
            #tax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

            #im = tax.imshow(values.transpose(), origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto')
            #im = tax.imshow(values)
            #tfig.colorbar(im, ax=tax, label='$\\Delta\\chi^2 -$ FC $\\chi^2$')

            # Plot and get contours
            #cs = ax.contour(10**xg, 10**yg, zg, levels=[4.61], colors=[color], linewidths=2)
            #cs = tax.contour(10**xg, 10**yg, zg, levels=[0], colors=[color], linewidths=2)
            #cs = tax.contourf(10**xg, 10**yg, zg, levels=[-10, 0, 100000], cmap='viridis', linewidths=2)

            #tax.set_xlim(0.01, 1)
            #tax.set_ylim(0.01, 100)

            #pp.savefig(tfig)
            #plt.close(tfig)
            #
            #contour_paths = []
            #for contour in cs.allsegs[0]:  
            #    if len(contour) > 1:  # Only keep meaningful contours
            #        contour_paths.append(contour)
            #
            #all_contours[hist_key] = contour_paths

    pp.close()
    pts /= count
    with uproot.recreate('pandora_brazil_pts.root') as file:
        file['pts'] = (pts, pts_x_edges, pts_y_edges)
        file['fcgrid'] = (fcgrid, pts_x_edges, pts_y_edges)
    x_centers = np.sqrt(pts_x_edges[:-1] * pts_x_edges[1:])
    y_centers = np.sqrt(pts_y_edges[:-1] * pts_y_edges[1:])
    
    interp = RegularGridInterpolator(
        (np.log10(x_centers), np.log10(y_centers)),
        #values - np.nanmin(values),
        pts,
        method='cubic',  
        bounds_error=False,
        fill_value=np.nan  
    )
     
    # Create grid
    N = 100
    x_min, x_max = np.log10(pts_x_edges[0]*1.05), np.log10(pts_x_edges[-1]*0.95)
    y_min, y_max = np.log10(pts_y_edges[0]*1.05), np.log10(pts_y_edges[-1]*0.95)
    xg = np.linspace(x_min, x_max, N)
    yg = np.linspace(y_min, y_max, N)
    xg_mesh, yg_mesh = np.meshgrid(xg, yg)

    # Interpolate
    zg = interp((xg_mesh, yg_mesh))
    zg = np.ma.masked_invalid(zg)
    
    #ls = [0.067, 0.159, 0.500, 0.841, 0.933]
    #ns = ['surf02', 'surf16', 'surf50', 'surf84', 'surf98']
    #cols = plt.cm.viridis(np.linspace(0, 1, len(ns)))
    #for n, l, c in zip(ns, ls, cols):
    #    # Plot and get contours
    #    cs = ax.contour(10**xg, 10**yg, zg, levels=[l], colors=[c], linewidths=2)
    #    
    #    contour_paths = []
    #    for contour in cs.allsegs[0]:  
    #        if len(contour) > 1:  # Only keep meaningful contours
    #            contour_paths.append(contour)
    #    
    #    all_contours[n] = list(reversed(sorted(contour_paths, key=lambda c: len(c))))

    all_contours = {
        n: sorted(filter(lambda c: len(c)>1,s), key=lambda c: -len(c)) 
        for n,s in zip(
            ['surf02', 'surf16', 'surf50', 'surf84', 'surf98'], 
            #ax.contour(10**xg, 10**yg, zg, levels=[0.067, 0.159, 0.500, 0.841, 0.933], linewidths=2).allsegs)
            #ax.contour(10**xg, 10**yg, zg, levels=[0.05, 0.159, 0.500, 0.841, 0.95], linewidths=2).allsegs)
            ax.contour(10**xg, 10**yg, zg, levels=[0.023, 0.159, 0.500, 0.841, 0.977], linewidths=2).allsegs)
    }

    ax.set_xlim(0.01, 1)
    #ax.set_ylim(1e-2, 100)
    ax.set_ylim(0.1, 100)
    #ax.set_xlim(0.05, 1)
    #ax.set_ylim(0.3, 100)
    plt.tight_layout()
    plt.show()
    
    return all_contours

def bootstrap_band(root_file, hist_pattern="surf"):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
    ax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

    all_contours = {}

    pts = None 
    allpts = []

    pts_x_edges = None
    pts_y_edges = None

    with uproot.open(fc_file) as fcfile:
        fcgrid = fcfile[fc_name].values()
        fcgrid = fcgrid.transpose()
    
    with uproot.open(root_file) as file:
        hist_keys = [key for key in file.keys() if key.startswith('brazil_throw')]
        
        for i in range(1000):
            keys = random.sample(hist_keys, k=1000)
            allpts.append(None)
            for hist_key in keys:
                hist = file[hist_key]
                values = hist.values()
                values = values - np.min(values)
                values = values - fcgrid
                #values = values - 4.61

                if allpts[i] is None:
                    allpts[i] = np.zeros(values.shape, dtype=float)

                allpts[i] += values >= 0
                x_edges = hist.axis(0).edges()
                y_edges = hist.axis(1).edges()
                if pts_x_edges is None: pts_x_edges = x_edges
                if pts_y_edges is None: pts_y_edges = y_edges
            allpts[i] /= 1000

    pts = sum(allpts) / len(allpts)
    x_centers = np.sqrt(pts_x_edges[:-1] * pts_x_edges[1:])
    y_centers = np.sqrt(pts_y_edges[:-1] * pts_y_edges[1:])
    
    interp = RegularGridInterpolator(
        (np.log10(x_centers), np.log10(y_centers)),
        #values - np.nanmin(values),
        pts,
        method='cubic',  
        bounds_error=False,
        fill_value=np.nan  
    )
     
    # Create grid
    N = 100
    x_min, x_max = np.log10(pts_x_edges[0]*1.05), np.log10(pts_x_edges[-1]*0.95)
    y_min, y_max = np.log10(pts_y_edges[0]*1.05), np.log10(pts_y_edges[-1]*0.95)
    xg = np.linspace(x_min, x_max, N)
    yg = np.linspace(y_min, y_max, N)
    xg_mesh, yg_mesh = np.meshgrid(xg, yg)

    # Interpolate
    zg = interp((xg_mesh, yg_mesh))
    zg = np.ma.masked_invalid(zg)
    
    #ls = [0.067, 0.159, 0.500, 0.841, 0.933]
    #ns = ['surf02', 'surf16', 'surf50', 'surf84', 'surf98']
    #cols = plt.cm.viridis(np.linspace(0, 1, len(ns)))
    #for n, l, c in zip(ns, ls, cols):
    #    # Plot and get contours
    #    cs = ax.contour(10**xg, 10**yg, zg, levels=[l], colors=[c], linewidths=2)
    #    
    #    contour_paths = []
    #    for contour in cs.allsegs[0]:  
    #        if len(contour) > 1:  # Only keep meaningful contours
    #            contour_paths.append(contour)
    #    
    #    all_contours[n] = list(reversed(sorted(contour_paths, key=lambda c: len(c))))

    all_contours = {
        n: sorted(filter(lambda c: len(c)>1,s), key=lambda c: -len(c)) 
        for n,s in zip(
            ['surf02', 'surf16', 'surf50', 'surf84', 'surf98'], 
            #ax.contour(10**xg, 10**yg, zg, levels=[0.067, 0.159, 0.500, 0.841, 0.933], linewidths=2).allsegs)
            #ax.contour(10**xg, 10**yg, zg, levels=[0.05, 0.159, 0.500, 0.841, 0.95], linewidths=2).allsegs)
            ax.contour(10**xg, 10**yg, zg, levels=[0.023, 0.159, 0.500, 0.841, 0.977], linewidths=2).allsegs)
    }

    ax.set_xlim(0.01, 1)
    #ax.set_ylim(1e-2, 100)
    ax.set_ylim(0.1, 100)
    #ax.set_xlim(0.05, 1)
    #ax.set_ylim(0.3, 100)
    plt.tight_layout()
    plt.show()
    
    return all_contours

def all_vs_exclusion(root_file, hist_pattern="surf"):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
    ax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

    all_contours = {}

    pts = None 
    pts_all = None
    count = 0
    count_all = 0

    pts_x_edges = None
    pts_y_edges = None

    with uproot.open(fc_file) as fcfile:
        fcgrid = fcfile[fc_name].values()
        fcgrid = fcgrid.transpose()
    
    pp = PdfPages(universes_file)
    with uproot.open(root_file) as file:
        #hist_keys = [key for key in file.keys() if key.startswith(hist_pattern)]
        hist_keys = [key for key in file.keys() if key.startswith('brazil_throw')]
        colors = plt.cm.viridis(np.linspace(0, 1, len(hist_keys)))

        for hist_key, color in zip(hist_keys, colors):
        #for hist_key in hist_keys:
            hist = file[hist_key]
            values = hist.values()
            values = values - np.min(values)
            values = values - fcgrid
            #values = values - 4.61

            if pts is None:
                pts = np.zeros(values.shape, dtype=float)
            if pts_all is None:
                pts_all = np.zeros(values.shape, dtype=float)

            if values[0,0] < 0:
                count += 1
                pts += values >= 0
            count_all += 1
            pts_all += values >= 0

#            print(f'{hist_key} \n\n {values}')
            x_edges = hist.axis(0).edges()
            y_edges = hist.axis(1).edges()
            if pts_x_edges is None: pts_x_edges = x_edges
            if pts_y_edges is None: pts_y_edges = y_edges
            
            # Calculate bin centers (log space)
            x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
            y_centers = np.sqrt(y_edges[:-1] * y_edges[1:])
            
            interp = RegularGridInterpolator(
                (np.log10(x_centers), np.log10(y_centers)),
                #values - np.nanmin(values),
                values,
                method='cubic',  
                bounds_error=False,
                fill_value=np.nan  
            )
             
            # Create grid
            N = 100
            x_min, x_max = np.log10(x_edges[0]*1.05), np.log10(x_edges[-1]*0.95)
            y_min, y_max = np.log10(y_edges[0]*1.05), np.log10(y_edges[-1]*0.95)
            xg = np.linspace(x_min, x_max, N)
            yg = np.linspace(y_min, y_max, N)
            xg_mesh, yg_mesh = np.meshgrid(xg, yg)

            # Interpolate
            zg = interp((xg_mesh, yg_mesh))
            zg = np.ma.masked_invalid(zg)
            
            #tfig, tax = plt.subplots(figsize=(7, 7), dpi=150)
            #tax.set(xscale='log', yscale='log')
            #tax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
            #tax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

            #im = tax.imshow(values.transpose(), origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto')
            #im = tax.imshow(values)
            #tfig.colorbar(im, ax=tax, label='$\\Delta\\chi^2 -$ FC $\\chi^2$')

            # Plot and get contours
            #cs = ax.contour(10**xg, 10**yg, zg, levels=[4.61], colors=[color], linewidths=2)
            #cs = tax.contour(10**xg, 10**yg, zg, levels=[0], colors=[color], linewidths=2)
            #cs = tax.contourf(10**xg, 10**yg, zg, levels=[-10, 0, 100000], cmap='viridis', linewidths=2)

            #tax.set_xlim(0.01, 1)
            #tax.set_ylim(0.1, 100)

            #pp.savefig(tfig)
            #plt.close(tfig)
            #
            #contour_paths = []
            #for contour in cs.allsegs[0]:  
            #    if len(contour) > 1:  # Only keep meaningful contours
            #        contour_paths.append(contour)
            #
            #all_contours[hist_key] = contour_paths

    pp.close()
    pts /= count
    pts_all /= count_all
    #with uproot.recreate('pandora_brazil_pts.root') as file:
    #    file['pts'] = (pts, pts_x_edges, pts_y_edges)
    #    file['fcgrid'] = (fcgrid, pts_x_edges, pts_y_edges)
    x_centers = np.sqrt(pts_x_edges[:-1] * pts_x_edges[1:])
    y_centers = np.sqrt(pts_y_edges[:-1] * pts_y_edges[1:])
    
    interp = RegularGridInterpolator(
        (np.log10(x_centers), np.log10(y_centers)),
        #values - np.nanmin(values),
        pts,
        method='cubic',  
        bounds_error=False,
        fill_value=np.nan  
    )
    interp_all = RegularGridInterpolator(
        (np.log10(x_centers), np.log10(y_centers)),
        #values - np.nanmin(values),
        pts_all,
        method='cubic',  
        bounds_error=False,
        fill_value=np.nan  
    )
     
    # Create grid
    N = 100
    x_min, x_max = np.log10(pts_x_edges[0]*1.05), np.log10(pts_x_edges[-1]*0.95)
    y_min, y_max = np.log10(pts_y_edges[0]*1.05), np.log10(pts_y_edges[-1]*0.95)
    xg = np.linspace(x_min, x_max, N)
    yg = np.linspace(y_min, y_max, N)
    xg_mesh, yg_mesh = np.meshgrid(xg, yg)

    # Interpolate
    zg = interp((xg_mesh, yg_mesh))
    zg = np.ma.masked_invalid(zg)

    zg_all = interp_all((xg_mesh, yg_mesh))
    zg_all = np.ma.masked_invalid(zg_all)
    
    #ls = [0.067, 0.159, 0.500, 0.841, 0.933]
    #ns = ['surf02', 'surf16', 'surf50', 'surf84', 'surf98']
    #cols = plt.cm.viridis(np.linspace(0, 1, len(ns)))
    #for n, l, c in zip(ns, ls, cols):
    #    # Plot and get contours
    #    cs = ax.contour(10**xg, 10**yg, zg, levels=[l], colors=[c], linewidths=2)
    #    
    #    contour_paths = []
    #    for contour in cs.allsegs[0]:  
    #        if len(contour) > 1:  # Only keep meaningful contours
    #            contour_paths.append(contour)
    #    
    #    all_contours[n] = list(reversed(sorted(contour_paths, key=lambda c: len(c))))

    all_contours = {
        n: sorted(filter(lambda c: len(c)>1,s), key=lambda c: -len(c)) 
        for n,s in zip(
            ['surf02', 'surf16', 'surf50', 'surf84', 'surf98'], 
            #ax.contour(10**xg, 10**yg, zg, levels=[0.067, 0.159, 0.500, 0.841, 0.933], linewidths=2).allsegs)
            #ax.contour(10**xg, 10**yg, zg, levels=[0.05, 0.159, 0.500, 0.841, 0.95], linewidths=2).allsegs)
            ax.contour(10**xg, 10**yg, zg, levels=[0.023, 0.159, 0.500, 0.841, 0.977], linewidths=2).allsegs)
    }
    all_contours_all = {
        n: sorted(filter(lambda c: len(c)>1,s), key=lambda c: -len(c)) 
        for n,s in zip(
            ['surf02', 'surf16', 'surf50', 'surf84', 'surf98'], 
            #ax.contour(10**xg, 10**yg, zg, levels=[0.067, 0.159, 0.500, 0.841, 0.933], linewidths=2).allsegs)
            #ax.contour(10**xg, 10**yg, zg, levels=[0.05, 0.159, 0.500, 0.841, 0.95], linewidths=2).allsegs)
            ax.contour(10**xg, 10**yg, zg_all, levels=[0.023, 0.159, 0.500, 0.841, 0.977], linewidths=2, linestyles='dashed').allsegs)
    }

    ax.set_xlim(0.01, 1)
    #ax.set_ylim(1e-2, 100)
    ax.set_ylim(0.1, 100)
    #ax.set_xlim(0.05, 1)
    #ax.set_ylim(0.3, 100)
    plt.tight_layout()
    plt.savefig("all_vs_exclusion.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    
    return all_contours

# Usage
#contour_data = plot_and_save_contours("pandora_brazil_surf.root")
#contour_data = plot_and_save_contours("pandora_brazil_small_surf.root")
#data_contour = make_data_contour("pandora_run2_surf.root")
#asimov_contour = make_data_contour("pandora_asimov_surf.root")
#contour_data = plot_and_save_contours("spine_brazil_surf.root")
#contour_data = plot_and_save_contours("spine_brazil_small_surf.root")
#data_contour = make_data_contour("spine_run2_surf.root")
#asimov_contour = make_data_contour("spine_asimov_surf.root")

contour_data = plot_and_save_contours(contour_file)
pandora_contour_data = plot_and_save_contours('pandora_brazil_4k_surf.root', 'pandora_90')
#contour_data = bootstrap_band(contour_file)
data_contour = make_data_contour(data_file)
#data_contour = make_fakedata_contour(contour_file)
asimov_contour = make_data_contour(asimov_file)

#all_vs_exclusion(contour_file)

def plot_contours_with_shading(contour_data):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set(xscale='log', yscale='log')
    ax.set_xlabel(r'$\sin^2 2 \theta_{\mu \mu}$', fontsize=20)
    ax.set_ylabel(r'$\Delta m_{41}^2$ [eV$^2$]', fontsize=20)

    # Extract contours in order
    contours = {
        #'outer_low': contour_data['surf02;1'][0],
        #'inner_low': contour_data['surf16;1'][0],
        #'middle': contour_data['surf50;1'][0],
        #'inner_high': contour_data['surf84;1'][0],
        #'outer_high': contour_data['surf98;1'][0],
        'outer_low': contour_data['surf02'][0],
        'inner_low': contour_data['surf16'][0],
        'middle': contour_data['surf50'][0],
        'inner_high': contour_data['surf84'][0],
        'outer_high': contour_data['surf98'][0],
        #'asimov': contour_data['surf;1'][0]
        'data': data_contour[0],
        'asimov': asimov_contour[0]
    }

    if add_external:
        for [name, file] in ext_data.items():
            if name == 'MINOS':
                with uproot.open(file[0]) as f:
                    contours[name] = f[file[1]].values()
            else: 
                contours[name] = np.genfromtxt(file, delimiter=',')
            

    # Shade between outer contours (2% and 98%)
    #ax.fill_betweenx(
    #    np.concatenate([contours['outer_low'][:,1], contours['outer_high'][:,1][::-1]]),
    #    np.concatenate([contours['outer_low'][:,0], contours['outer_high'][:,0][::-1]]),
    #    color='yellow', alpha=0.6, label='+/-2σ'
    #)

    if add_external:
        ax.plot(contours['MiniBooNE'][:,0], contours['MiniBooNE'][:,1],
                color='#E69F00', alpha=0.6, linewidth=2, linestyle='--', label='MiniBooNE 90%')
        ax.plot(contours['CCFR'][:,0], contours['CCFR'][:,1],
                color='#56B4E9', alpha=0.6, linewidth=2, linestyle='--', label='CCFR 90%')
        ax.plot(contours['CDGS'][:,0], contours['CDGS'][:,1],
                color='#CC79A7',alpha=0.6,  linewidth=2, linestyle='--', label='CDGS 90%')
        ax.plot(np.sin(np.arcsin(np.sqrt(contours['NOvA'][:,0]))*2.0)**2, contours['NOvA'][:,1],
                color='#0072B2', alpha=0.6, linewidth=2, linestyle='--', label='NOvA 90%')
        ax.plot(np.sin(np.arcsin(np.sqrt(contours['MINOS'][0]))*2.0)**2, contours['MINOS'][1],
                color='#009E73', alpha=0.6, linewidth=2, linestyle='--', label='MINOS+ 90%')
        #ax.plot(contours['IceCube'][:,0], contours['IceCube'][:,1],
        #        color='#D55E00', linewidth=2, linestyle='-')
        ax.fill(contours['IceCube'][:,0],contours['IceCube'][:,1], edgecolor='#D55E00', facecolor='#D55E00', linewidth=1, linestyle="-",alpha=0.6, label='IceCube 95%')
        ax.plot([1.587e-1],[3.535], '*', color='#D55E00', markersize=12, label='IceCube Best Fit')
        #ax.fill_betweenx(
        #    np.concatenate([contours['outer_low'][:,1], contours['outer_high'][:,1][::-1]]),
        #    np.concatenate([contours['outer_low'][:,0], contours['outer_high'][:,0][::-1]]),
        #    x2=1,
        #    color='gray', alpha=0.6, label='+/-2σ'
        #)


        ## Shade between inner contours (16% and 84%)
        #ax.fill_betweenx(
        #    np.concatenate([contours['inner_low'][:,1], contours['inner_high'][:,1][::-1]]),
        #    np.concatenate([contours['inner_low'][:,0], contours['inner_high'][:,0][::-1]]),
        #    x2=1,
        #    color='#009E73', alpha=1, label='+/-1σ'
        #)

        #ax.plot(contours['data'][:,0], contours['data'][:,1], 
        #        color='black', linewidth=2, label=data_label)
        ax.plot(contours['middle'][:,0], contours['middle'][:,1], 
                color='black', linewidth=2, linestyle='solid', label='ICARUS SPINE 90% Median Exclusion')
        ax.plot(pandora_contour_data['surf50'][0][:,0], pandora_contour_data['surf50'][0][:,1], 
                color='gray', linewidth=2, linestyle='solid', label='ICARUS Pandora 90% Median Exclusion')
    else:
        ax.fill_betweenx(
            np.concatenate([contours['outer_low'][:,1], contours['outer_high'][:,1][::-1]]),
            np.concatenate([contours['outer_low'][:,0], contours['outer_high'][:,0][::-1]]),
            x2=1,
            color='#F0E442', alpha=0.6, label='+/-2σ'
        )


        # Shade between inner contours (16% and 84%)
        ax.fill_betweenx(
            np.concatenate([contours['inner_low'][:,1], contours['inner_high'][:,1][::-1]]),
            np.concatenate([contours['inner_low'][:,0], contours['inner_high'][:,0][::-1]]),
            x2=1,
            color='#009E73', alpha=1, label='+/-1σ'
        )
        # Plot the middle contour in black
        ax.plot(contours['middle'][:,0], contours['middle'][:,1], 
                color='black', linewidth=2, linestyle='--', label='Median Exclusion')
        ax.plot(contours['asimov'][:,0], contours['asimov'][:,1], 
                color='#56B4E9', linewidth=2, linestyle='--', label='Asimov Sensitivity')
        #ax.plot(contours['data'][:,0], contours['data'][:,1], 
        #        color='black', linewidth=2, label=data_label)
        ax.plot([], [], ' ', label='ICARUS Run 2 90% CL')
    #ax.plot(contours['data'][:,0], contours['data'][:,1], 
    #        color='black', linewidth=2, label='Fake Data')

    # Add legend
    ax.legend(fontsize=12, frameon=False, reverse=True, loc='lower left')

    ax.text(0.625, 100.0, 'PROfit')

    return fig, ax


fig, ax = plot_contours_with_shading(contour_data)
ax.set_xlim(0.003, 1)
#ax.set_xlim(0.03, 1)
#ax.set_ylim(1e-2, 80)
ax.set_ylim(0.01, 90)
#ax.set_ylim(0.1, 90)
#ax.set_xlim(0.05, 1)
#ax.set_ylim(0.3, 100)
plt.tight_layout()
plt.savefig(fig_name, format="pdf", bbox_inches='tight')
plt.show()
