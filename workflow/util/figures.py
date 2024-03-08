import os
from pathlib import Path
import sys
import glob
from os.path import join
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio 
import rasterio.plot
import rasterio.mask
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mpc
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection

def plot_risk_disadv(ens_plot, comm_list, comm_names, comm_titles,
                     tracts, save_filename, risk_cmap='Reds', bounds=None):
    """
    Take a geodataframe that contains
    a risk burden column and plot this
    in a municipality with shading for
    4 relevant disadvantaged community status
    indicators. 

    ens_plot: GeoDataFrame with risk burden column
    comm_list: list of GeoDataFrames that represent the
    disadvantaged communities
    comm_names: list of column names for each entry in comm_list
    comm_titles: Dictionary of community references to 
    subplot titles
    tracts: GeoDataFrame of census tract boundaries
    save_filename: str for saving file
    risk_cmap: named colormap, default Reds
    bounds: list of minx, maxx, miny, maxy, default None
    """

    fig, ax = plt.subplots(figsize=(7, 6),
                           nrows=2,
                           ncols=2,
                           dpi=600,
                           gridspec_kw = {'wspace':.01,'hspace': .22})
    
    # Get rel_eal_pct column for plotting
    ens_plot['rel_eal_plot'] = np.round(ens_plot['rel_eal']*100)

    # Loop through each layer
    for i, comm_geo in enumerate(comm_list):
        row = i%2
        col = i//2
        
        # Reproject layer to tract crs
        comm_plot = comm_geo.to_crs(tracts.crs)
        
        # Layer name
        name_l = comm_names[i]

        # Plot the layer boundary in each column
        comm_plot.plot(ax=ax[row, col],
                       color='#18437f',
                       alpha=.3,
                       edgecolor='none')

        # Plot census tract boundaries in each column
        tracts.plot(ax=ax[row, col],
                    edgecolor='black',
                    color='none',
                    lw=.5)

        # Caption the subplot title
        ax[row, col].set_title(comm_titles[name_l],
                               size=12)
        
        # Plot the eal
        vmin_eal = 0
        vmax_eal = ens_plot['rel_eal_plot'].max()
        # Sort before plotting
        ens_plot = ens_plot.sort_values('rel_eal_plot', ascending=True)
        ens_plot.plot(ax=ax[row, col],
                      column='rel_eal_plot',
                      cmap=risk_cmap,
                      s=1,
                      vmin=vmin_eal,
                      vmax=vmax_eal)

        # Set axis off but keep outline
        ax[row, col].tick_params(axis='both',
                                 which='both',
                                 bottom=False,
                                 left=False,
                                 labelbottom=False,
                                 labelleft=False)

        # Zoom in on points
        if bounds is not None:
            minx = bounds[0]
            miny = bounds[1]
            maxx = bounds[2]
            maxy = bounds[3]
            ax[row, col].set_xlim([minx, maxx])
            ax[row, col].set_ylim([miny, maxy])

        # Display a legend, just to make it clear
        # which areas are "disadvantaged"
        legend_elements = [Patch(facecolor='#18437f',
                                 alpha=.3,
                                 label='Disadvantaged\nCommunity')]
        
        ax[row, col].legend(handles=legend_elements,
                            loc='center',
                            fontsize='small',
                            bbox_to_anchor=(.73, .9),
                            frameon=False)
    
    # Add continuous legends
    cax = fig.add_axes([.21, 0.06, .61, 0.02])
    sm = plt.cm.ScalarMappable(cmap=risk_cmap,
                              norm=plt.Normalize(vmin=vmin_eal,
                                                 vmax=vmax_eal))
    sm._A = []
    cbr = fig.colorbar(sm, cax=cax, orientation='horizontal',
                       format=ticker.PercentFormatter(decimals=0))
    cbr.ax.tick_params(labelsize=12) 
    cbr.set_label("Risk Burden (%)", size=12)

    fig.savefig(save_filename,
                dpi=600,
                bbox_inches='tight')

def plot_alloc(ens_plot, elev_ids,
               scen, budget, sort_cols,
               title_dict, save_filename,
               size=15, alpha=.5):
    """
    Take a dataframe that contains
    a risk burden column and a structure value column
    and plot the homes that are elevated or not for a
    given budget according to various funding rules.

    ens_plot: GeoDataFrame with risk burden column
    elev_ids: a dictionary of scenario/policy keys to
    fd_id list of all elevated homes for that given
    scenario/policy
    scen: The hazard/damage scenario
    budget: The allocation amount considered for plotting
    sort_cols: The funding rules (part of a policy)
    title_dict: Mapping funding rules to subplot titles
    save_filename: str for saving file
    size: markersize of points
    alpha: shading of non elevated points
    """

    fig, ax = plt.subplots(figsize=(8, 10),
                           nrows=3,
                           ncols=2,
                           dpi=600,
                           sharey=True,
                           sharex=True,
                           gridspec_kw = {'wspace':.15,'hspace': .35})

    # Some preprocessing to help the plotting look nicer
    # get structural value in thousands
    ens_plot['val_s_thou'] = ens_plot['val_s']/1e3
    # get the rel_eal_plot columns
    ens_plot['rel_eal_plot'] = np.round(ens_plot['rel_eal']*100)

    for i, sort in enumerate(sort_cols):
        col = i % 2
        row = i // 2

        # Find the policy
        policy = sort + '_' + str(budget)
        fd_ids = elev_ids[scen + '_' + policy]

        # Subset into homes that are elevated for this policy
        elevated = ens_plot.loc[ens_plot['fd_id'].isin(fd_ids)]
        not_elev = ens_plot.loc[~ens_plot['fd_id'].isin(fd_ids)]

        # Plot the points
        # We'll use the size/2 so that the points
        # which aren't elevated don't show as prominently
        # We'll also plot them first so that they are below the
        # elevated points
        not_elev.plot(x='val_s_thou',
                      y='rel_eal_plot',
                      kind='scatter',
                      edgecolor='gray',
                      color='white',
                      alpha=alpha,
                      label='Not Elevated',
                      s=size/2,
                      ax=ax[row, col])
        # We'll plot these in the green #225522 so they stand out
        elevated.plot(x='val_s_thou',
                      y='rel_eal_plot',
                      kind='scatter',
                      color='#225522',
                      label='Elevated',
                      s=size,
                      ax=ax[row, col])
        
        # Add title
        ax[row, col].set_title(title_dict[sort],
                            size=12)
        
        # Clean up the plot
        # Remove left ticks from each axis in column 1
        # ax[row, 1].tick_params(axis='both',
        #                        which='both',
        #                        left=False)
        # Remove bottom ticks from each axis in row 1
        ax[0, col].tick_params(axis='both',
                               which='both')

        # Update labels
        ax[row, 0].set_ylabel('Risk Burden (%)',
                            size=12)
        ax[row, 0].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ax[2, col].set_xlabel('Structure Value in $1,000s',
                              size=12)

    ax[0, 1].get_legend().remove()
    ax[1, 1].get_legend().remove()
    ax[1, 0].get_legend().remove()
    ax[2, 1].get_legend().remove()
    ax[2, 0].get_legend().remove()

    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[0, 0].legend(handles[::-1], labels[::-1],
                    ncols=1)

    fig.savefig(save_filename,
                dpi=600,
                bbox_inches='tight')
    
def plot_objcst(obj_plot, obj_cols, obj_names,
                scen, color_dict, min_obj_cols,
                avoid_eq_col,
                save_filename):
    """
    Take a dataframe that contains
    the objective evaluation for different
    policies. This function plots each objective
    value as a function of the upfront cost for
    a given modeling scenario, which the user
    passes in. Currently, this is over-defined
    for our particular case study. 

    obj_plot: DataFrame of all evaluations
    obj_cols: list of the 4 objectives (very specific to this study)
    obj_names: list of the readable names of the objectives
    scen: str, the modeling scenario
    color_dict: dict, for funding rule hues
    min_obj_cols: list of axes to reverse
    avoid_eq_col: str, name of how inequity in
    investment objective is calculated
    save_filename: str, filepath for saving
    """
    
    objs = obj_plot[obj_plot['scen'] == scen]

    # Prep plot colums
    objs['cost_plot'] = objs['up_cost']/1e6
    objs['npv_plot'] = objs['npv']/1e6
    objs['pv_resid_plot'] = objs['pv_resid']/1e6
    
    # We want to plot the single j40 points
    j40 = objs[objs['sort'] == 'cejst']
    # In our case study, j40 has negative npv
    # because of the distribution of risk burden. 
    # Therefore, we can represent a single point
    # for visual purposes (this is only relevant
    # for Upper scenarios where you can
    # invest in more elevation with 
    # larger budgets) as the one with
    # the highest npv. This corresponds to
    # the lowest upfront cost. 
    j40_plot = j40.loc[j40['npv'].idxmax()]
    j40_avoid_eq = j40_plot[avoid_eq_col]
    j40_resid = j40_plot['pv_resid_plot']
    j40_npv = j40_plot['npv_plot']
    j40_resid_eq = j40_plot['resid_eq']
    j40_cost = j40_plot['cost_plot']

    objs = objs[objs['sort'] != 'cejst']

    fig, ax = plt.subplots(figsize=(8, 6),
                       nrows=2,
                       ncols=2,
                       sharex=True,
                       dpi=600)

    # Plot the justice40 points
    ax[0, 0].plot(j40_cost, j40_npv,
                c=color_dict['cejst'],
                marker='s',
                markersize=8,
                zorder=0)
    ax[0, 1].plot(j40_cost, j40_resid,
                c=color_dict['cejst'],
                marker='s',
                markersize=8,
                zorder=0)
    ax[1, 0].plot(j40_cost, j40_avoid_eq,
                c=color_dict['cejst'],
                marker='s',
                markersize=8,
                zorder=0)
    ax[1, 1].plot(j40_cost, j40_resid_eq,
                c=color_dict['cejst'],
                marker='s',
                markersize=8,
                zorder=0)

    # Loop through each value in 'sort'
    # Plot the kde 
    # Solid lines community, dashed for household
    # Unique colors
    lw=2

    for i, obj in enumerate(obj_cols):
        row = i // 2
        col = i % 2

        # Plot boxplots for each policy besides Justice40
        sns.lineplot(x='cost_plot',
                    y=obj,
                    data=objs[objs['res'] == 'household'],
                    hue='sort',
                    palette=color_dict,
                    lw=lw,
                    legend=False,
                    ax=ax[row, col])
        sns.lineplot(data=objs[objs['res'] == 'community'],
                    x='cost_plot',
                    y=obj,
                    hue='sort',
                    lw=lw,
                    ls='dashed',
                    palette=color_dict,
                    legend=False,
                    ax=ax[row, col])
        
        ax[row, col].set_xlabel('Upfront Cost ($M)', size=12)

        # Set y label
        ax[row, col].set_ylabel(obj_names[i], size=14)

        # These are minimize objectives, so reverse y-axis
        if obj in min_obj_cols:
            rect = Rectangle([0, ax[row, col].get_ylim()[0]],
                        6, ax[row, col].get_ylim()[1])
            ax[row, col].set_ylim(ax[row, col].get_ylim()[::-1])

        else:
            height = ax[row, col].get_ylim()[1]
            if ax[row, col].get_ylim()[0] < 0:
                height = height + 0 - ax[row, col].get_ylim()[0]
            rect = Rectangle([0, ax[row, col].get_ylim()[0]],
                        6, height)
        
        pc = PatchCollection([rect], facecolor='grey', alpha=0.2)
        ax[row, col].add_collection(pc)

        # Arrow for direction of improvement
        ax[row, col].annotate('Direction of\nImprovement',
                    xy=(.68, .08), xycoords='axes fraction',
                    xytext=(0.68, .08), textcoords='axes fraction',
                    fontsize=8,
                    color='black',
                    horizontalalignment='left',
                    verticalalignment='center')
        ax[row, col].annotate("", xytext=(.66, 0.02), xy=(.66, .15),
                            xycoords='axes fraction',
                            textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="simple",
                                    color='black'))
    fig.tight_layout()
    fig.align_ylabels(ax[:, 1])

    # Add legend below
    legend_elements = [Line2D([0], [0], color='black', ls='none',
                            label='Household Criteria'),
                    Line2D([0], [0], color=color_dict['npv_opt'], lw=lw,
                            label='Net Benefit'),
                    Line2D([0], [0], color=color_dict['avoid_rel_eal'], lw=lw,
                            label='Risk Burden Reduction'),
                    Line2D([0], [0], color=color_dict['rel_eal'], lw=lw,
                            label='Initial Risk Burden'),
                    Line2D([0], [0], color='black', ls='none',
                            label='Community Criteria'),
                    Line2D([0], [0], color=color_dict['lmi'], lw=lw,
                            ls='dashed',
                            label='Low-Mod Income'),
                    Line2D([0], [0], color=color_dict['ovb'], lw=lw,
                            ls='dashed',
                            label='NJ Overburdened'),
                    Line2D([0], [0], color=color_dict['cejst'],
                            label='Justice40 (FEMA Now)',
                            markersize=8,
                            ls='none', marker='s')]

    ax[1, 0].legend(handles=legend_elements,
                    loc='center',
                    ncol = 2,
                    fontsize='large',
                    bbox_to_anchor=(1.03, -.45))
    
    fig.savefig(save_filename,
                dpi=600,
                bbox_inches='tight')