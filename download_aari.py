if __name__ == '__main__':
    with open('1.txt','a') as f:
        for year in range(2018,2022):
            for month in range(1,13):
                if month < 10:
                    month1 = str(0) + str(month)
                else:
                    month1 = str(month)
                for day in range(1, 32):
                    if day < 10:
                        day1 = str(0) + str(day)
                    else:
                        day1 = str(day)
                    f.write(r'http://wdc.aari.ru/datasets/d0015/arcice/' + str(year) + r'/' +  r'aari_arc_'
                            +str(year)+str(month1)+str(day1)+'_pl_a.zip' + '\n')


