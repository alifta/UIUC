# Paths



def dataset_db(file_input=['data'], file_output=['uiuc.db'], output=True):
    """
    Read log files of each user, detect user_hash based on the log names
    then rename the folder to the user_id and extract data from log files
    """

    # Paths
    DATA = os.path.join(ROOT_PATH, file_input[0])
    DB = os.path.join(DB_PATH, file_output[0])

    # Reading files
    files = dir_walk(DATA)

    # Dictionary {User : MAC}
    user_mac = {}
    for f in files:
        sp = f.split('/')
        # Second to last is user_id [1-28]
        u = int(sp[-2])
        # Last is the file_name = datetime + mac
        d, mac = sp[-1].split('.')
        dict_add(user_mac, u, mac)

    # Sort user_mac based on user_id (key)
    user_mac = sorted([(int(k), v) for k, v in user_mac.items()])

    # Save user_mac to DB
    query = f'''
    CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    mac TEXT NOT NULL
    );
    '''
    db_create(DB, query)
    query = 'INSERT INTO users VALUES (?,?)'
    db_insert_many(DB, query, user_mac)

    # Read BT and WiFI data
    # error_list = [] # TEST
    user_b = defaultdict(list)  # Bluetooth
    user_w = defaultdict(list)  # WiFi
    b_mac = []  # Bluetooth MAC list
    w_mac = []  # WiFi MAC list

    # Add BT users to the list of bluetooth devices
    # They are not needed in WiFi
    # Because they only see access points MAC in logs
    b_mac = [u[1] for u in user_mac]

    for file in files:
        is_b = False
        is_w = False
        sp = file.split('/')
        u = int(sp[-2])
        d, mac = sp[-1].split('.')
        # Check the type of log file
        if d[0] == 's':  # BT
            is_b = True
        else:  # WiFi
            is_w = True

        # Read file line by line -> save to DF -> convert DF column to series
        df = pd.read_csv(file, names=['line'])
        lines = df.line
        last_time = ''

        for i in range(len(lines)):
            # Length of line with time info = 21
            # Length of line with mac info = 40
            if len(lines[i]) < 40:  # We read a line with time info ...
                # String format
                # last_time = lines[i]
                # Datetime format
                last_time = datetime.strptime(lines[i][2:], '%m-%d-%Y %H:%M:%S')
                continue
            else:  # We read a line with mac address ...
                # If last_time is out of range i.e. < 2010-2-25 then skip until read a valid one
                if last_time < datetime(2010, 2, 25):
                    continue
                # If the last_time was valid and the line is a BT or wifi do ...
                if is_b:  # Bluetooth
                    b_index = 0
                    # b_index = -1  # TEST: changed to -1 to detect the possible errors
                    if lines[i] not in b_mac:
                        # Add new MAC to list
                        b_mac.append(lines[i])
                        # Index of new mac = (len - 1) of list, becasue it was the latest added element
                        b_index = len(b_mac) - 1
                    else:
                        b_index = b_mac.index(lines[i])
                    # Add new BT log to dictionary {user:[(time,mac)]
                    user_b[u].append((last_time, b_index))
                    # TEST
                    # if b_index < 0:
                    # if last_time < datetime(2010,2,24):
                    # error_list.append((file,lines[i],last_time))

                else:  # WiFi
                    w_index = 0
                    # w_index = -1  # TEST
                    if lines[i] not in w_mac:
                        # Add new MAC to list
                        w_mac.append(lines[i])
                        # Index of new mac = (len - 1) of list becasue it is the last added element
                        w_index = len(w_mac) - 1
                    else:
                        w_index = w_mac.index(lines[i])
                    user_w[u].append((last_time, w_index))
                    # TEST
                    # if w_index < 0:
                    # if last_time < datetime(2010,2,24):
                    # error_list.append((file,lines[i],last_time))
    # TEST : save the errors
    # np.savetxt('error_list.csv', error_list, delimiter=',', fmt='%s')

    if output:
        print('# BT devices:', len(b_mac))
        print('# WiFi devices:', len(w_mac))

    # Save {BT_id : BT_MAC} to DB
    query = f'''
    CREATE TABLE IF NOT EXISTS bluetooth (
    id INTEGER PRIMARY KEY,
    mac TEXT NOT NULL
    );
    '''
    db_create(DB, query)
    b_mac_insert = [(i, b_mac[i]) for i in range(len(b_mac))]
    query = 'INSERT INTO bluetooth VALUES (?,?)'
    db_insert_many(DB, query, b_mac_insert)

    # Save {WiFi_id : WiFi_MAC} to DB
    query = f'''
    CREATE TABLE IF NOT EXISTS wifi (
    id INTEGER PRIMARY KEY,
    mac TEXT NOT NULL
    );
    '''
    db_create(DB, query)
    w_mac_insert = [(i, w_mac[i]) for i in range(len(w_mac))]
    query = 'INSERT INTO wifi VALUES (?,?)'
    db_insert_many(DB, query, w_mac_insert)

    # Save BT and WiFi logs to DB with following columns
    # user_node, bluetooth_node / ap_node , time, bluetooth = 0 / wifi = 1
    query = f'''
    CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY,
    user INTEGER NOT NULL,
    mac INTEGER NOT NULL,
    time TIMESTAMP,
    wifi INTEGER
    );
    '''
    db_create(DB, query)
    interaction_insert = []
    for k, v in user_b.items():  # k = user
        # v is a list of interactions = (time, MAC)
        for item in v:
            interaction_insert.append((k, item[1], item[0], 0))
    for k, v in user_w.items():
        for item in v:
            interaction_insert.append((k, item[1], item[0], 1))
    query = 'INSERT INTO logs(user,mac,time,wifi) VALUES (?,?,?,?)'
    db_insert_many(DB, query, interaction_insert)

    if output:
        print('# interactions:', len(interaction_insert))