import ast
import glob
import json
import os
import pickle
import random
import re
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import combinations, chain
from collections import Counter, defaultdict

import matplotlib
import networkx as nx
import reverse_geocode

import pandas as pd
import numpy as np
import yaml

from geopy.distance import EARTH_RADIUS, great_circle
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from intervaltree import IntervalTree
from tqdm import tqdm

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['data_dir']


NETWORK_FEATURE_NAMES = ['node', 'log_in_degree', 'log_shared_neighbors', 'log_forward_weight',
                         'log_reverse_weight', 'send_recency', 'receive_recency', 'reverse_recency', 'forward_recency']


def pad_jagged_array(jagged, padding=-1):
    max_len = max(len(r) for r in jagged)
    array = np.full((len(jagged), max_len), padding)

    for i, row in enumerate(jagged):
        array[i, :len(row)] = row

    return array


def plot_map(file_name):
    with open(file_name, 'rb') as f:
        lats, lons = pickle.load(f)

    lats, lons = lats[1:], lons[1:]
    ax = plt.axes(projection=ccrs.GOOGLE_MERCATOR)

    img_extent = (-97.94, -97.56, 30.09, 30.52)
    img = plt.imread('austin.png')

    print()
    ax.set_extent(img_extent, crs=ccrs.Geodetic())
    ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())

    trans = ccrs.GOOGLE_MERCATOR.transform_points(ccrs.Geodetic(), lons, lats)

    plt.scatter(trans[:, 0], trans[:, 1], c='red', transform=ccrs.GOOGLE_MERCATOR, marker='x', alpha=0.3)
    # plt.scatter(trans[:, 0], trans[:, 1], c='red', transform=ccrs.GOOGLE_MERCATOR, marker='.', s=1)

    plt.savefig('checkin-map.png', bbox_inches='tight', dpi=2000)
    # plt.show()
    # plt.close()


class DatasetParser(ABC):
    item_names = []
    name = None
    description = None
    citation = None
    item_feature_names = []
    item_features = 0
    source = ''
    categorical_item_features = []
    categorical_chooser_features = []
    item_id_feature = None

    @classmethod
    @abstractmethod
    def load(cls):
        ...


class ItemFeatureDatasetParser(DatasetParser, ABC):
    @classmethod
    def write(cls):
        choice_sets, item_df, choices, person_df, choosers = cls.load()
        m = len(choice_sets)
        n = len(choice_sets[0])

        out_dir = f'data/{cls.name}'
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(f'{out_dir}/meta', exist_ok=True)

        with open(f'{out_dir}/choices.txt', 'w') as f:
            f.write('sample chooser item chosen\n' +
                    '\n'.join(
                        f'{row} {choosers[row]} {choice_sets[row, item]} {int(choices[row] == choice_sets[row, item])}'
                        for row in range(m)
                        for item in range(n)
                        if choice_sets[row, item] >= 0))

        item_df.to_csv(f'{out_dir}/items.txt', sep=' ', index_label='item', float_format='%g')

        if person_df is not None:
            person_df.to_csv(f'{out_dir}/choosers.txt', sep=' ', index_label='chooser')

        choice_set_lengths = (choice_sets >= 0).sum(1)
        metadata = {'name': cls.name,
                    'varied_choice_set_size': len(np.unique(choice_set_lengths)) > 1,
                    'max_choice_set_size': int(max(choice_set_lengths)),
                    'repeat_choices': len(choosers) != len(np.unique(choosers)),
                    'unique_items': len(item_df.index),
                    'item_features': cls.item_features,
                    'chooser_features': len(person_df.columns) if person_df is not None else False,
                    'samples': len(choice_sets),
                    'description': cls.description,
                    'citation': cls.citation,
                    'source': cls.source,
                    'categorical_chooser_features': cls.categorical_chooser_features,
                    'categorical_item_features': cls.categorical_item_features,
                    'item_id_feature': cls.item_id_feature}

        with open(f'{out_dir}/meta/info.json', 'w') as f:
            json.dump(metadata, f)


class USElection2016Parser(ItemFeatureDatasetParser):
    name = 'election-2016'
    description = 'This is a county-level voting dataset.'
    citation = """"""

    source = 'https://github.com/mkearney/presidential_election_county_results_2016'
    categorical_item_features = []
    categorical_chooser_features = ['Rural-urban_Continuum_Code_2013', 'Economic_typology_2015']

    @classmethod
    def load(cls):
        vote_df = pd.read_csv(
            f'{DATA_DIR}/usa-2016-election/presidential_election_county_results_2016/data/pres.elect16.results.2018.csv')
        vote_df = vote_df[vote_df.county.notnull()]

        # for fips, county in vote_df.groupby('fips'):
        #     trump_votes = county[county['cand'] == 'Donald Trump'].votes.values[0]
        #     clinton_votes = county[county['cand'] == 'Hillary Clinton'].votes.values[0]
        #     other_votes = county[~county['cand'].isin(['Donald Trump', 'Hillary Clinton'])]
        #
        #     if len(other_votes[other_votes.votes > trump_votes]) > 0:
        #         print(fips, other_votes[other_votes.votes > trump_votes])
        #
        #     if len(other_votes[other_votes.votes > clinton_votes]) > 0:
        #         print(fips, other_votes[other_votes.votes > clinton_votes])
        #     # print(trump_votes, clinton_votes, other_votes)

        all_fips = vote_df.fips.unique().astype(int)
        all_cands = vote_df.cand.unique()

        with open(f'{DATA_DIR}/usa-2016-election/gnn-residual-correlation-election/adjacency.txt', 'r') as f:
            adjacency = [x.rstrip().split('\t') for x in f.readlines()]

        # Build county adjacency graph
        edges = ''
        current_fips = None
        for i in range(len(adjacency)):
            if adjacency[i][0] != '':
                current_fips = adjacency[i][1]
                continue

            if adjacency[i][3] != current_fips:
                edges += f'{current_fips} {adjacency[i][3]}\n'

        sci_df = pd.read_csv(
            f'{DATA_DIR}/usa-2016-election/gnn-residual-correlation-election/county_county_aug2020.tsv', delimiter='\t')
        sci_df = sci_df[sci_df['user_loc'] != sci_df['fr_loc']]
        social_edges = sci_df.loc[sci_df.groupby('user_loc')['scaled_sci'].nlargest(10).index.get_level_values(1)][
            ['user_loc', 'fr_loc']].values

        # Build county social graph
        edges = ''
        current_fips = None
        for i in range(len(adjacency)):
            if adjacency[i][0] != '':
                current_fips = adjacency[i][1]
                continue

            if adjacency[i][3] != current_fips:
                edges += f'{current_fips} {adjacency[i][3]}\n'

        catgorical_cols = ['Rural-urban_Continuum Code_2013', 'Economic_typology_2015']
        use_cols = ['Rural-urban_Continuum Code_2013', 'Economic_typology_2015', 'R_death_2016', 'R_birth_2016',
                    'R_NET_MIG_2016', 'BachelorRate2016', 'MedianIncome2016', 'Unemployment_rate_2016']

        edu_df = pd.read_csv(f'{DATA_DIR}/usa-2016-election/gnn-residual-correlation-election/education.csv').set_index(
            'FIPS')
        inc_df = pd.read_csv(f'{DATA_DIR}/usa-2016-election/gnn-residual-correlation-election/income.csv',
                             thousands=',').set_index('FIPS')
        pop_df = pd.read_csv(
            f'{DATA_DIR}/usa-2016-election/gnn-residual-correlation-election/population.csv').set_index('FIPS')
        emp_df = pd.read_csv(
            f'{DATA_DIR}/usa-2016-election/gnn-residual-correlation-election/unemployment.csv').set_index('FIPS').drop(
            columns=['State', 'Area_name'])

        int_cols = catgorical_cols + ['MedianIncome2016']
        chooser_df = edu_df.join([inc_df, pop_df, emp_df]).loc[all_fips, use_cols]
        chooser_df = chooser_df.astype({col: 'int64' for col in int_cols}).rename(columns=lambda x: x.replace(' ', '_'))

        cand_to_idx = {cand: i for i, cand in enumerate(all_cands)}
        vote_df['cand'] = vote_df['cand'].map(cand_to_idx)

        counts = []
        choice_sets = []
        choices = []
        choosers = []
        for fips, county in vote_df.groupby('fips'):

            ballot = county.cand.to_numpy()

            for cand, votes in county[['cand', 'votes']].values:
                if votes > 0:
                    counts.append(votes)
                    choice_sets.append(ballot)
                    choices.append(cand)
                    choosers.append(fips)

        choice_sets = pad_jagged_array(choice_sets)

        item_df = pd.DataFrame([(i, cand) for cand, i in cand_to_idx.items()], columns=['candidate', 'name']).set_index(
            'candidate')

        out_dir = f'data/{cls.name}'
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/chooser-graph.txt', 'w') as f:
            f.write(edges)

        np.savetxt(f'{out_dir}/social-edges.txt', social_edges.astype(int), fmt='%i')

        with open(f'{out_dir}/choice-counts.txt', 'w') as f:
            f.write('sample count\n' + '\n'.join(f'{sample} {count}' for sample, count in enumerate(counts)))

        return choice_sets, item_df, choices, chooser_df, choosers


class CaliforniaElectionParser(ItemFeatureDatasetParser):
    name = ''
    description = 'This is a precint-level CA voting dataset.'
    citation = """"""

    source = 'https://statewidedatabase.org/election.html'
    categorical_item_features = []
    categorical_chooser_features = []

    vote_fname = ''
    reg_fname = ''
    adj_fname = ''
    codebook_dir = ''

    @classmethod
    def load(cls):
        vote_df = pd.read_csv(cls.vote_fname, dtype=str).apply(pd.to_numeric, errors='ignore')
        reg_df = pd.read_csv(cls.reg_fname, dtype=str).apply(pd.to_numeric, errors='ignore')

        # elec_types = {'PR_', 'PRS', 'CNG', 'ASS', 'USS', 'SEN'}
        # districted_elec_types = {'CNG', 'ASS', 'SEN'}
        # district_map = {'CNG': 'CDDIST', 'ASS': 'ADDIST', 'SEN': 'SDDIST'}
        # elec_regex = re.compile(r'(SEN|CNG|USS|PRS|PR_|ASS)([0-9]*)(.*)')

        # Without assembly and president
        elec_types = {'PR_', 'CNG', 'USS', 'SEN'}
        districted_elec_types = {'CNG', 'SEN'}
        district_map = {'CNG': 'CDDIST', 'SEN': 'SDDIST'}
        elec_regex = re.compile(r'(SEN|CNG|USS|PR_)([0-9]*)(.*)')

        cand_name_map = dict()
        for fname in glob.glob(f'{cls.codebook_dir}/*'):
            with open(fname, 'r') as f:
                lines = [line.split('\t') for line in f.readlines()]

            for line in lines:
                seat, cand = line[0], line[1].strip('*')
                matches = elec_regex.match(seat)
                if matches is not None:
                    cand_name_map[seat] = cand

        adj_df = pd.read_csv(cls.adj_fname, delimiter='\t', dtype=str)

        common_srprecs = set(vote_df.SRPREC_KEY.unique()).intersection(reg_df.SRPREC_KEY.unique()).intersection(
            adj_df.SRPREC_KEY.unique())

        vote_df = vote_df[vote_df.SRPREC_KEY.isin(common_srprecs)]
        reg_df = reg_df[reg_df.SRPREC_KEY.isin(common_srprecs)]
        adj_df = adj_df[adj_df.SRPREC_KEY.isin(common_srprecs)]

        vote_cols = {col for col in vote_df.columns if elec_regex.match(col) is not None}

        cand_to_item_id = {cand: i for i, cand in enumerate(cand_name_map.keys())}

        chooser_df = reg_df.set_index('SRPREC_KEY')

        # Aggregate over asian countries
        asian_countries = ['KOR', 'JPN', 'CHI', 'IND', 'VIET', 'FIL']
        chooser_df['ASNDEM'] = chooser_df[[f'{x}DEM' for x in asian_countries]].sum(axis=1)
        chooser_df['ASNREP'] = chooser_df[[f'{x}REP' for x in asian_countries]].sum(axis=1)
        chooser_df = chooser_df.drop(columns=[f'{x}{y}' for x in asian_countries for y in ('DEM', 'REP')])

        # Aggregate over M/F and DEM/REP for per-age groups
        for age in ['1824', '2534', '3544', '4554', '5564', '65PL']:
            chooser_df[f'{age}'] = chooser_df[[f'{party}{x}{age}' for x in ('M', 'F') for party in ('DEM', 'REP')]].sum(axis=1)
            chooser_df = chooser_df.drop(columns=[f'{party}{x}{age}' for x in ('M', 'F') for party in ('DEM', 'REP')])

        # Aggregate over DEM/REP for ethnicity info
        for ethnicity in ['HISP', 'JEW', 'ASN']:
            chooser_df[f'{ethnicity}'] = chooser_df[[f'{ethnicity}{party}' for party in ('DEM', 'REP')]].sum(axis=1)
            chooser_df = chooser_df.drop(columns=[f'{ethnicity}{party}' for party in ('DEM', 'REP')])

        # https://statewidedatabase.org/info/metadata/SOR_codebook.html
        chooser_feat_names = [
            'DEM', 'REP', 'AIP', 'PAF', 'LIB', 'GRN', 'MALE', 'FEMALE', 'HISP', 'JEW', 'ASN',
            '1824', '2534', '3544', '4554', '5564', '65PL'
        ]

        chooser_df = chooser_df[chooser_feat_names].div(chooser_df.TOTREG_R, axis='index')

        counts = []
        choice_sets = []
        choices = []
        choosers = []

        for row in tqdm(vote_df.itertuples(), total=len(vote_df)):
            cands_by_seat = defaultdict(list)
            counts_by_seat = defaultdict(list)

            for col in vote_cols:
                elec_type = col[:3]

                cand_id = col
                if elec_type in districted_elec_types:
                    cand_id = f'{elec_type}{str(getattr(row, district_map[elec_type])).zfill(2)}{col[3:]}'

                if cand_id in cand_name_map:
                    # Handle districtless elections
                    if elec_type == 'PRS' or elec_type == 'USS':
                        cands_by_seat[elec_type].append(cand_id)
                        counts_by_seat[elec_type].append(getattr(row, col))
                    else:
                        cands_by_seat[cand_id[:5]].append(cand_id)
                        counts_by_seat[cand_id[:5]].append(getattr(row, col))

            for seat in cands_by_seat.keys():
                choice_set = [cand_to_item_id[cand] for cand in cands_by_seat[seat]]

                # Add choice for each candidate running for each seat
                for i in range(len(choice_set)):
                    choice_sets.append(choice_set)
                    counts.append(counts_by_seat[seat][i])
                    choices.append(choice_set[i])
                    choosers.append(row.SRPREC_KEY)

        choice_sets = pad_jagged_array(choice_sets)

        item_df = pd.DataFrame([(i, cand) for cand, i in cand_to_item_id.items()], columns=['item', 'name']).set_index(
            'item')

        out_dir = f'data/{cls.name}'
        os.makedirs(out_dir, exist_ok=True)
        edges = ''
        for row in adj_df.itertuples():
            edges += '\n'.join(f'{row.SRPREC_KEY} {dest}' for dest in row.nbrs.split(',')) + '\n'

        with open(f'{out_dir}/chooser-graph.txt', 'w') as f:
            f.write(edges)

        with open(f'{out_dir}/choice-counts.txt', 'w') as f:
            f.write('sample count\n' + '\n'.join(f'{sample} {count}' for sample, count in enumerate(counts)))

        return choice_sets, item_df, choices, chooser_df, choosers


class CaliforniaElection2020Parser(CaliforniaElectionParser):
    name = 'ca-election-2020'

    vote_fname = f'{DATA_DIR}/california-2020-election/state_g20_sov_data_by_g20_srprec.csv'
    reg_fname = f'{DATA_DIR}/california-2020-election/state_g20_registration_by_g20_srprec.csv'
    adj_fname = f'{DATA_DIR}/california-2020-election/g20-precinct-adjacency.txt'
    codebook_dir = f'{DATA_DIR}/california-2020-election/g20-county-codebooks'


class CaliforniaElection2016Parser(CaliforniaElectionParser):
    name = 'ca-election-2016'

    vote_fname = f'{DATA_DIR}/california-2016-election/state_g16_sov_data_by_g16_srprec/state_g16_sov_data_by_g16_srprec.csv'
    reg_fname = f'{DATA_DIR}/california-2016-election/state_g16_registration_by_g16_srprec/state_g16_registration_by_g16_srprec.csv'
    adj_fname = f'{DATA_DIR}/california-2016-election/g16-precinct-adjacency.txt'
    codebook_dir = f'{DATA_DIR}/california-2016-election/g16-county-codebooks'


class FriendsAndFamilyAppUsageParser(ItemFeatureDatasetParser):
    item_feature_names = ['app', 'recency']
    item_features = len(item_feature_names)

    name = 'app-usage'
    description = 'This is an Android app usage dataset.'
    citation = """"""

    source = 'http://realitycommons.media.mit.edu/friendsdataset.html'
    categorical_item_features = []
    categorical_chooser_features = []

    item_id_feature = 'app'

    @classmethod
    def load(cls):

        app_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/App.csv')

        participants = app_df.participantID.unique()

        common_apps = set(app_df[app_df.participantID == 'fa10-01-01'].apppackage)

        for name, group in app_df.groupby('participantID'):
            common_apps = common_apps.intersection(group.apppackage)

        # Exclude built-ins
        exclude = ['com.android', 'com.motorola', 'com.htc', 'com.sec', 'com.google']
        common_apps = common_apps.union(
            [app for app in app_df.apppackage.unique() if any(app.startswith(x) for x in exclude)])

        running_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/AppRunning.csv')

        # Just look at choices in March and April
        running_df['scantime'] = pd.to_datetime(running_df['scantime'])
        running_df = running_df[(running_df['scantime'].dt.month == 4) | (running_df['scantime'].dt.month == 3)]


        # Remove apps owned by < 10 people or owned by everyone:
        app_df = app_df[app_df.groupby('apppackage')['participantID'].transform('nunique') >= 10]
        app_df = app_df[~app_df.apppackage.isin(common_apps)]
        running_df = running_df[running_df.package.isin(app_df.apppackage.unique())]

        # Construct intervals containing installed apps
        app_df = app_df.sort_values(by=['scantime'])
        app_df['scantime'] = pd.to_datetime(app_df['scantime'])

        prev_installed = {participant: set() for participant in participants}
        curr_installed = {participant: set() for participant in participants}
        curr_time = {participant: None for participant in participants}
        interval_start_time = {participant: datetime.min for participant in participants}

        installed_apps = {participant: IntervalTree() for participant in participants}
        for row in tqdm(app_df.itertuples(), total=len(app_df)):
            time = row.scantime
            participant = row.participantID
            app = row.apppackage

            if time != curr_time[participant]:
                if curr_installed[participant] != prev_installed[participant]:
                    installed_apps[participant][interval_start_time[participant]:time] = curr_installed[participant]

                    prev_installed[participant] = curr_installed[participant]
                    interval_start_time[participant] = time
                    curr_installed[participant] = set()
                    curr_time[participant] = time

            if row.appuninstalled == 'no':
                curr_installed[participant].add(app)

        # Close all intervals
        for participant in participants:
            installed_apps[participant][interval_start_time[participant]:datetime.max] = curr_installed[participant]

        # count_df = running_df.package.value_counts().rename_axis('apppackage').reset_index(name='counts')

        running_df = running_df.sort_values(by=['scantime'])
        running_df['scantime'] = pd.to_datetime(running_df['scantime'])

        last_used_times = {participant: dict() for participant in participants}

        items = []
        choosers = []
        choice_sets = []
        choices = []

        item = 0

        use_time_updates = dict()
        prev_time = None

        for row in tqdm(running_df.itertuples(), total=len(running_df)):
            participant = row.participantID
            time = row.scantime
            app = row.package

            # print(participant, time, app)

            # Only update use_times when we tick over to a new time to prevent 0 time since used for simultaneous usages
            if time != prev_time:
                # print('NEW TIME!. Updating last used times.')
                for participant, app in use_time_updates:
                    last_used_times[participant][app] = prev_time
                use_time_updates = dict()

            # if app in last_used_times[participant]:
            #     print('last usage:', (time - last_used_times[participant][app]).seconds, 'seconds ago')
            # else:
            #     print('first usage!')

            import warnings
            warnings.filterwarnings("error")

            if app not in last_used_times[participant] or (time - last_used_times[participant][app]).seconds > 60 * 60:
                if (participant, app) not in use_time_updates:
                    # print(f'new choice!')
                    choice_set = list(installed_apps[participant][time].pop().data.union({app}))
                    choice = choice_set.index(app)

                    choice_sets.append(np.arange(item, item + len(choice_set)))
                    choices.append(item + choice)
                    choosers.append(participant)

                    item += len(choice_set)

                    # print(choice_set)
                    # print([(time - last_used_times[participant][x]).seconds if x in last_used_times[participant] else None for x in choice_set])
                    try:
                        items.extend(np.column_stack((
                            choice_set,
                            [(1 / np.log(max((time - last_used_times[participant][x]).seconds, 60 * 60))) if x in last_used_times[participant] else 0 for
                             x in choice_set]
                        )))
                    except RuntimeWarning:
                        print(time, app, participant)
                        print(choice_set)
                        print([(time - last_used_times[participant][x]).seconds if x in last_used_times[participant] else None for x in choice_set])
                        print(use_time_updates)
                        exit()
                # else:
                #     print('Already handled this timestamp, skipping')

            use_time_updates[participant, app] = time
            prev_time = time

            # print()

        item_df = pd.DataFrame(items, columns=['app', 'recency'])
        choice_sets = pad_jagged_array(choice_sets)

        proximity_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/BluetoothProximity.csv').dropna()
        proximity_df['date'] = pd.to_datetime(proximity_df['date'])

        # Find pairs with top 5 proximities
        pair_cols = ['participantID', 'participantID.B']

        edges = ''
        for participant in proximity_df['participantID'].unique():
            pairs = proximity_df[proximity_df['participantID'] == participant][pair_cols].apply(tuple,
                                                                                                axis='columns').values
            counts = Counter(pairs)
            edges += '\n'.join(f'{participant} {pair[1]}' for pair, count in counts.most_common(10)) + '\n'
        # friendship_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/SurveyFriendship.csv')
        # edges = '\n'.join(f'{s} {t}' for s, t in friendship_df[['source', 'target']].values) + '\n'

        out_dir = f'data/{cls.name}'
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/chooser-graph.txt', 'w') as f:
            f.write(edges)

        return choice_sets, item_df, choices, None, choosers



        # dates = sorted({date for participant in participants for date in date_counts[participant].keys()})
        # counts = {participant: [date_counts[participant][date] for date in dates] for participant in participants}

        # print(dates)
        # print(counts)

        # plt.figure(figsize=(10, 5))
        #
        # for participant in participants:
        #     plt.plot(dates, counts[participant], alpha=0.5)
        #
        # plt.gcf().autofmt_xdate()
        #
        # plt.ylabel('App Usages')
        # plt.savefig('plots/app_usage_counts.pdf', bbox_inches='tight')
        # plt.close()


        # count_df = count_df[~count_df.apppackage.isin(common_apps)]
        #
        # with pd.option_context('display.max_rows', None, 'display.max_columns',None):
        #
        #     print(count_df)
        #
        #     # print(running_df.groupby('package')['participantID'].nunique().sort_values(ascending=False).reset_index(name='count'))


class FriendsAndFamilyAppInstallationParser(ItemFeatureDatasetParser):
    item_feature_names = ['app']
    item_features = len(item_feature_names)

    name = 'app-install'
    description = 'This is an Android app installation dataset.'
    citation = """"""

    source = 'http://realitycommons.media.mit.edu/friendsdataset.html'
    categorical_item_features = []
    categorical_chooser_features = []

    item_id_feature = 'app'

    @classmethod
    def load(cls):

        app_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/App.csv')

        participants = app_df.participantID.unique()

        common_apps = set(app_df[app_df.participantID == 'fa10-01-01'].apppackage)

        for name, group in app_df.groupby('participantID'):
            common_apps = common_apps.intersection(group.apppackage)

        # Exclude built-ins
        exclude = ['com.android', 'com.motorola', 'com.htc', 'com.sec', 'com.google']
        common_apps = common_apps.union([app for app in app_df.apppackage.unique() if any(app.startswith(x) for x in exclude)])

        print('Common:\n', common_apps)

        # Remove apps only owned by >=10 person or owned by everyone:
        app_df = app_df[app_df.groupby('apppackage')['participantID'].transform('nunique') >= 10]

        app_df = app_df[~app_df.apppackage.isin(common_apps)]

        app_df = app_df.sort_values(by=['scantime'])
        app_df['scantime'] = pd.to_datetime(app_df['scantime'])

        prev_installed = {participant: set() for participant in participants}
        curr_installed = {participant: set() for participant in participants}
        curr_time = {participant: None for participant in participants}

        choosers = []
        already_installed = []
        choices = []

        for row in tqdm(app_df.itertuples(), total=len(app_df)):
            time = row.scantime
            participant = row.participantID
            app = row.apppackage

            if time != curr_time[participant]:
                if curr_installed[participant] != prev_installed[participant]:
                    for app in curr_installed[participant].difference(prev_installed[participant]):
                        choosers.append(participant)
                        already_installed.append(prev_installed[participant])
                        choices.append(app)

                    prev_installed[participant] = prev_installed[participant].union(curr_installed[participant])
                    curr_installed[participant] = set()
                    curr_time[participant] = time

            curr_installed[participant].add(app)

        print('Installs:', len(choices))
        print('Unique apps:', len(np.unique(choices)))

        unique_apps = set(choices)

        item_to_id = {app: i for i, app in enumerate(unique_apps)}

        choices = [item_to_id[app] for app in choices]
        choice_sets = [[item_to_id[app] for app in unique_apps.difference(current_apps)] for current_apps in already_installed]

        print('choice set lens:', [len(x) for x in choice_sets])
        choice_sets = pad_jagged_array(choice_sets)

        item_df = pd.DataFrame(sorted(unique_apps, key=lambda app: item_to_id[app]), columns=['app'])

        proximity_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/BluetoothProximity.csv').dropna()
        proximity_df['date'] = pd.to_datetime(proximity_df['date'])

        # Find pairs with top 5 proximities
        pair_cols = ['participantID', 'participantID.B']

        edges = ''
        for participant in proximity_df['participantID'].unique():
            pairs = proximity_df[proximity_df['participantID'] == participant][pair_cols].apply(tuple, axis='columns').values
            counts = Counter(pairs)
            edges += '\n'.join(f'{participant} {pair[1]}' for pair, count in counts.most_common(10)) + '\n'

        # friendship_df = pd.read_csv(f'{DATA_DIR}/friends-and-family/SurveyFriendship.csv')
        # edges = '\n'.join(f'{s} {t}' for s, t in friendship_df[['source', 'target']].values) + '\n'

        out_dir = f'data/{cls.name}'
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/chooser-graph.txt', 'w') as f:
            f.write(edges)

        return choice_sets, item_df, choices, None, choosers


if __name__ == '__main__':


    # CaliforniaElection2016Parser.write()
    # CaliforniaElection2020Parser.write()
    # USElection2016Parser.write()


    FriendsAndFamilyAppInstallationParser.write()
    FriendsAndFamilyAppUsageParser.write()

