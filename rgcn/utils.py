"""
Utility functions for link prediction


"""
import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict
import torch
import pandas as pd
import calendar
from math import radians,sin,cos,degrees,atan2
import time
import math
import execjs
import random
from collections import OrderedDict
import csv


# ===== Dataset auto switch for loc-grid =====
_CURRENT_DATASET = None

_DATASET_GRID_CFG = {
    "ICEWS14s":   {"grid_start_id": 7617,  "locgrid_csv": "locgrid_one2one_level7-14s.csv"},
    "ICEWS05-15": {"grid_start_id": 10978, "locgrid_csv": "locgrid_one2one_level7-15.csv"},
    "ICEWS18":    {"grid_start_id": 23523, "locgrid_csv": "locgrid_one2one_level7-18.csv"},
}

_LOCGRID_DF_CACHE = {}  # csv_path -> dataframe


def _norm_dataset_name(name: str) -> str:
    if name is None:
        return ""
    name = name.strip()
    # 兼容 ICEWS18+* 这类命名
    name = name.replace("+", "").replace("*", "")
    return name


def _set_current_dataset(name: str):
    global _CURRENT_DATASET
    _CURRENT_DATASET = _norm_dataset_name(name)


def _get_grid_cfg():
    key = _norm_dataset_name(_CURRENT_DATASET)
    return _DATASET_GRID_CFG.get(key, None)


def _resolve_csv_path(filename: str) -> str:
    import os
    cand = [
        filename,
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "..", filename),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    return filename  # fallback


def _get_locgrid_df(csv_path: str):
    if csv_path not in _LOCGRID_DF_CACHE:
        _LOCGRID_DF_CACHE[csv_path] = pd.read_csv(csv_path, sep="\t")
    return _LOCGRID_DF_CACHE[csv_path]

ctx = execjs.compile('''    


//已改
    function encode_geosot_2d(level, lat, lon) {
        let lat_deg_min_sec = get_deg_min_sec(lat);
        let lon_deg_min_sec = get_deg_min_sec(lon);

        let lat_deg_bin = lat_deg_min_sec[0].toString(2);
        let lon_deg_bin = lon_deg_min_sec[0].toString(2);//直接把分钟的数字转化为二进制就是GeoSOT的经度/纬度方向编码了
        lat_deg_bin = Array(9 - lat_deg_bin.length).join('0') + lat_deg_bin;
        lon_deg_bin = Array(9 - lon_deg_bin.length).join('0') + lon_deg_bin;//前几位补0，一共补到二进制九位

        let lat_min_bin = lat_deg_min_sec[1].toString(2);
        let lon_min_bin = lon_deg_min_sec[1].toString(2);
        lat_min_bin = Array(7 - lat_min_bin.length).join('0') + lat_min_bin;
        lon_min_bin = Array(7 - lon_min_bin.length).join('0') + lon_min_bin;

        let lat_sec_bin = Math.floor(lat_deg_min_sec[2] / LON_LAT_DMS_TABLE[level]).toString(2);
        let lon_sec_bin = Math.floor(lon_deg_min_sec[2] / LON_LAT_DMS_TABLE[level]).toString(2);
		if(level>15){
			//b0 = Array(0).join('0');   -->  ''
			//b1 = Array(1).join('0');   -->  ''
			//b2 = Array(2).join('0');   -->  '0'
			//b3 = Array(3).join('0');   -->  '00'
			//应该是这块的问题，有时候补0补少了
			lat_sec_bin = Array(level + 1 - 15 - lat_sec_bin.length).join('0') + lat_sec_bin;
			lon_sec_bin = Array(level + 1 - 15 - lon_sec_bin.length).join('0') + lon_sec_bin;
		}
		else{
			lat_sec_bin = "";
			lon_sec_bin = "";
		}
        let lat_bin = lat_deg_bin.substr(0, 9) + lat_min_bin.substr(0, 7) + lat_sec_bin;
        let lon_bin = lon_deg_bin.substr(0, 9) + lon_min_bin.substr(0, 7) + lon_sec_bin;
        let lat_cap = lat >= 0 ? '0' : '1';
        let lon_cap = lon >= 0 ? '0' : '1';

        lat_bin = lat_cap + lat_bin;
        lon_bin = lon_cap + lon_bin;
        let code_array = Array();
        for (var i = 0; i < level; i++) {
            code_array.push(lat_bin[i]);
            code_array.push(lon_bin[i]);
        }
        return code_array.join('');
    }
		//已改
	function encode_geosot_3d(level, lat, lon, height) {
        let lat_deg_min_sec = get_deg_min_sec(lat);
        let lon_deg_min_sec = get_deg_min_sec(lon);

        let lat_deg_bin = lat_deg_min_sec[0].toString(2);
        let lon_deg_bin = lon_deg_min_sec[0].toString(2);//直接把分钟的数字转化为二进制就是GeoSOT的经度/纬度方向编码了
        lat_deg_bin = Array(9 - lat_deg_bin.length).join('0') + lat_deg_bin;
        lon_deg_bin = Array(9 - lon_deg_bin.length).join('0') + lon_deg_bin;//前几位补0，一共补到二进制九位

        let lat_min_bin = lat_deg_min_sec[1].toString(2);
        let lon_min_bin = lon_deg_min_sec[1].toString(2);
        lat_min_bin = Array(7 - lat_min_bin.length).join('0') + lat_min_bin;
        lon_min_bin = Array(7 - lon_min_bin.length).join('0') + lon_min_bin;

        let lat_sec_bin = Math.floor(lat_deg_min_sec[2] / LON_LAT_DMS_TABLE[level]).toString(2);
        let lon_sec_bin = Math.floor(lon_deg_min_sec[2] / LON_LAT_DMS_TABLE[level]).toString(2);
		if(level>15){
			lat_sec_bin = Array(level + 1 - 15 - lat_sec_bin.length).join('0') + lat_sec_bin;
			lon_sec_bin = Array(level + 1 - 15 - lon_sec_bin.length).join('0') + lon_sec_bin;
		}
		else{
			lat_sec_bin = "";
			lon_sec_bin = "";
		}
        let lat_bin = lat_deg_bin.substr(0, 9) + lat_min_bin.substr(0, 7) + lat_sec_bin;
        let lon_bin = lon_deg_bin.substr(0, 9) + lon_min_bin.substr(0, 7) + lon_sec_bin;
		let height_bin;
		if (height>0){
			height_bin = Math.floor(Math.abs(height) / HEIGHT_TABLE[level]).toString(2);
		}
		else{
			height_bin = Math.floor(Math.abs(height) / HEIGHT_TABLE_UNDERGROUND[level]).toString(2);
		}
        height_bin = Array(level - height_bin.length).join('0') + height_bin;

        let lat_cap = lat >= 0 ? '0' : '1';
        let lon_cap = lon >= 0 ? '0' : '1';
        let height_cap = height >= 0 ? '0' : '1';

        lat_bin = lat_cap + lat_bin;
        lon_bin = lon_cap + lon_bin;
        height_bin = height_cap + height_bin;
        let code_array = Array();
        for (var i = 0; i < level; i++) {
            code_array.push(lat_bin[i]);
            code_array.push(lon_bin[i]);
            code_array.push(height_bin[i]);
        }
        return code_array.join('');
    }




	//前几位是度（大于180、360的取回180、360），中间几位取出来转为整数的分（大于60的取回60），后边转为整数的秒
	//这里边对应的是二维编码
	//已改
    function decode_geosot_2d(code) {
        console.assert(code.length % 2 === 0 && code.length > 2);
        let level = code.length / 2;
		part_label = 0;
        let latArray_degree = Array();
        let lonArray_degree = Array();
        let latArray_minute = Array();
        let lonArray_minute = Array();
        let latArray_second = Array();
        let lonArray_second = Array();
        let heightArray = Array();
		//注意这块儿是从2开始的，不是从0开始的
        for (var i = 2; i < code.length; i++) {
            if (i % 2 === 0) {
				if(part_label<9-1){
					latArray_degree.push(code[i]);
				}
				else if(part_label<15-1){
					latArray_minute.push(code[i]);
				}
				else{
					latArray_second.push(code[i]);
				}
            }
            else if (i % 2 === 1) {
				if(part_label<9-1){
					lonArray_degree.push(code[i]);
				}
				else if(part_label<15-1){
					lonArray_minute.push(code[i]);
				}
				else{
					lonArray_second.push(code[i]);
				}
				part_label = part_label+1;
            }
        }
        let latString_degree = latArray_degree.join('');
        let lonString_degree = lonArray_degree.join('');
        let latString_minute = latArray_minute.join('');
        let lonString_minute = lonArray_minute.join('');
        let latString_second = latArray_second.join('');
        let lonString_second = lonArray_second.join('');
        let heightString = heightArray.join('');
		let latInt_degree,lonInt_degree,latInt_minute,lonInt_minute,latInt_second,lonInt_second;
		if (lonString_degree.length>0){
			latInt_degree = parseInt(latString_degree, 2);
			lonInt_degree = parseInt(lonString_degree, 2);
		}
		else{
			latInt_degree = 0;
			lonInt_degree = 0;
		}
		if (lonString_minute.length>0){
			latInt_minute = parseInt(latString_minute, 2);
			lonInt_minute = parseInt(lonString_minute, 2);
		}
		else{
			latInt_minute = 0;
			lonInt_minute = 0;
		}
		if (lonString_second.length>0){
			latInt_second = parseInt(latString_second, 2);
			lonInt_second = parseInt(lonString_second, 2);
		}
		else{
			latInt_second = 0;
			lonInt_second = 0;
		}
		
		latminute_part_min = latInt_minute * LON_LAT_DMS_TABLE[level<15?level:15]>60?60:latInt_minute * LON_LAT_DMS_TABLE[level<15?level:15];
		lonminute_part_min = lonInt_minute * LON_LAT_DMS_TABLE[level<15?level:15]>60?60:lonInt_minute * LON_LAT_DMS_TABLE[level<15?level:15];
		latsecond_part_min = latInt_second * LON_LAT_DMS_TABLE[level]>60?60:latInt_second * LON_LAT_DMS_TABLE[level];
		lonsecond_part_min = lonInt_second * LON_LAT_DMS_TABLE[level]>60?60:lonInt_second * LON_LAT_DMS_TABLE[level];
		
		let unlatMin = latInt_degree * LON_LAT_DMS_TABLE[level<9?level:9] + latminute_part_min / 60 + latsecond_part_min / 3600;
        let unlonMin = lonInt_degree * LON_LAT_DMS_TABLE[level<9?level:9] + lonminute_part_min / 60 + lonsecond_part_min / 3600;
        let unlatMax,unlonMax;
		if (level<=9){
			//unlonMax 这些的里的un是没有符号的意思
		unlatMax = (latInt_degree+1) * LON_LAT_DMS_TABLE[level];
		unlonMax = (lonInt_degree+1) * LON_LAT_DMS_TABLE[level];
		unlatMax = unlatMax<90?unlatMax:90;		
		unlonMax = unlonMax<180?unlonMax:180;
		}
		
		
		else if (level<=15){
		let latminute_part_max = (latInt_minute+1) * LON_LAT_DMS_TABLE[level]>60? 60: (latInt_minute+1) * LON_LAT_DMS_TABLE[level];
		let lonminute_part_max = (lonInt_minute+1) * LON_LAT_DMS_TABLE[level]>60? 60: (lonInt_minute+1) * LON_LAT_DMS_TABLE[level];
		unlatMax = latInt_degree * LON_LAT_DMS_TABLE[9] + latminute_part_max / 60;
		unlonMax = lonInt_degree * LON_LAT_DMS_TABLE[9] + lonminute_part_max / 60;
		}
		
		
		else{
		let latsecond_part_max = (latInt_second+1) * LON_LAT_DMS_TABLE[level]>60? 60: (latInt_second+1) * LON_LAT_DMS_TABLE[level];
		let lonsecond_part_max = (lonInt_second+1) * LON_LAT_DMS_TABLE[level]>60? 60: (lonInt_second+1) * LON_LAT_DMS_TABLE[level];
		unlatMax = latInt_degree * LON_LAT_DMS_TABLE[9] + latInt_minute*LON_LAT_DMS_TABLE[15] / 60 + latsecond_part_max/3600
		unlonMax = lonInt_degree * LON_LAT_DMS_TABLE[9] + lonInt_minute*LON_LAT_DMS_TABLE[15] / 60 + lonsecond_part_max/3600;
		}
		
        let latMin = code[0] === '1' ? -unlatMax : unlatMin;
		let lonMin = code[1] === '1' ? -unlonMax : unlonMin;
        let latMax = code[0] === '1' ? -unlatMin : unlatMax;
		let lonMax = code[1] === '1' ? -unlonMin : unlonMax;
		
		
        latMin = latMin < -90 ? -90 : latMin;
		lonMin = lonMin < -180 ? -180 : lonMin;
        latMax = latMax > 90 ? 90 : latMax;
        lonMax = lonMax > 180 ? 180: lonMax;
		
        return {
            level: level,
            min_coord: [latMin, lonMin],
            max_coord: [latMax, lonMax]
        }
    }
	//前几位是度（大于180、360的取回180、360），中间几位取出来转为整数的分（大于60的取回60），后边转为整数的秒
	//已改
    function decode_geosot_3d(code) {
        console.assert(code.length % 3 === 0 && code.length > 3);
        let level = code.length / 3;
		part_label = 0;
        let latArray_degree = Array();
        let lonArray_degree = Array();
        let latArray_minute = Array();
        let lonArray_minute = Array();
        let latArray_second = Array();
        let lonArray_second = Array();
        let heightArray = Array();
		//注意这块儿是从3开始的，不是从0开始的
        for (var i = 3; i < code.length; i++) {
            if (i % 3 === 0) {
				if(part_label<9-1){
					latArray_degree.push(code[i]);
				}
				else if(part_label<15-1){
					latArray_minute.push(code[i]);
				}
				else{
					latArray_second.push(code[i]);
				}
            }
            else if (i % 3 === 1) {
				if(part_label<9-1){
					lonArray_degree.push(code[i]);
				}
				else if(part_label<15-1){
					lonArray_minute.push(code[i]);
				}
				else{
					lonArray_second.push(code[i]);
				}
				part_label = part_label+1;
            }
            else {
                heightArray.push(code[i]);
            }
        }
        let latString_degree = latArray_degree.join('');
        let lonString_degree = lonArray_degree.join('');
        let latString_minute = latArray_minute.join('');
        let lonString_minute = lonArray_minute.join('');
        let latString_second = latArray_second.join('');
        let lonString_second = lonArray_second.join('');
        let heightString = heightArray.join('');
		let latInt_degree,lonInt_degree,latInt_minute,lonInt_minute,latInt_second,lonInt_second;
		if (lonString_degree.length>0){
			latInt_degree = parseInt(latString_degree, 2);
			lonInt_degree = parseInt(lonString_degree, 2);
		}
		else{
			latInt_degree = 0;
			lonInt_degree = 0;
		}
		if (lonString_minute.length>0){
			latInt_minute = parseInt(latString_minute, 2);
			lonInt_minute = parseInt(lonString_minute, 2);
		}
		else{
			latInt_minute = 0;
			lonInt_minute = 0;
		}
		if (lonString_second.length>0){
			latInt_second = parseInt(latString_second, 2);
			lonInt_second = parseInt(lonString_second, 2);
		}
		else{
			latInt_second = 0;
			lonInt_second = 0;
		}
		
		latminute_part_min = latInt_minute * LON_LAT_DMS_TABLE[level<15?level:15]>60?60:latInt_minute * LON_LAT_DMS_TABLE[level<15?level:15];		
		lonminute_part_min = lonInt_minute * LON_LAT_DMS_TABLE[level<15?level:15]>60?60:lonInt_minute * LON_LAT_DMS_TABLE[level<15?level:15];
		latsecond_part_min = latInt_second * LON_LAT_DMS_TABLE[level]>60?60:latInt_second * LON_LAT_DMS_TABLE[level];
		lonsecond_part_min = lonInt_second * LON_LAT_DMS_TABLE[level]>60?60:lonInt_second * LON_LAT_DMS_TABLE[level];
		
        let unlatMin = latInt_degree * LON_LAT_DMS_TABLE[level<9?level:9] + latminute_part_min / 60 + latsecond_part_min / 3600;
        let unlonMin = lonInt_degree * LON_LAT_DMS_TABLE[level<9?level:9] + lonminute_part_min / 60 + lonsecond_part_min / 3600;
        let unlatMax,unlonMax;
		if (level<=9){
			//unlonMax 这些的里的un是没有符号的意思
		unlatMax = (latInt_degree+1) * LON_LAT_DMS_TABLE[level];	
		unlonMax = (lonInt_degree+1) * LON_LAT_DMS_TABLE[level];
		unlatMax = unlatMax<90?unlatMax:90;	
		unlonMax = unlonMax<180?unlonMax:180;
		}
		
		
		else if (level<=15){
		let latminute_part_max = (latInt_minute+1) * LON_LAT_DMS_TABLE[level]>60? 60: (latInt_minute+1) * LON_LAT_DMS_TABLE[level];
		let lonminute_part_max = (lonInt_minute+1) * LON_LAT_DMS_TABLE[level]>60? 60: (lonInt_minute+1) * LON_LAT_DMS_TABLE[level];
		unlatMax = latInt_degree * LON_LAT_DMS_TABLE[9] + latminute_part_max / 60;
		unlonMax = lonInt_degree * LON_LAT_DMS_TABLE[9] + lonminute_part_max / 60;
		}
		
		
		else{
		let latsecond_part_max = (latInt_second+1) * LON_LAT_DMS_TABLE[level]>60? 60: (latInt_second+1) * LON_LAT_DMS_TABLE[level];
		let lonsecond_part_max = (lonInt_second+1) * LON_LAT_DMS_TABLE[level]>60? 60: (lonInt_second+1) * LON_LAT_DMS_TABLE[level];
		unlatMax = latInt_degree * LON_LAT_DMS_TABLE[9] + latInt_minute*LON_LAT_DMS_TABLE[15] / 60 + latsecond_part_max/3600
		unlonMax = lonInt_degree * LON_LAT_DMS_TABLE[9] + lonInt_minute*LON_LAT_DMS_TABLE[15] / 60 + lonsecond_part_max/3600;
		}
		
        let latMin = code[0] === '0' ? unlatMin : -unlatMax;
		let lonMin = code[1] === '0' ? unlonMin : -unlonMax;
        let latMax = code[0] === '0' ? unlatMax : -unlatMin;
		let lonMax = code[1] === '0' ? unlonMax : -unlonMin;
		
		let heightInt = parseInt(heightString, 2);
        let heightResolution = HEIGHT_TABLE[level];
        let heightResolution_Underground = HEIGHT_TABLE_UNDERGROUND[level];
		
		//再次修改
		let heightMin = code[2] === '0' ? heightInt * heightResolution : -heightInt * heightResolution_Underground - heightResolution_Underground;
        let heightMax = code[2] === '0' ? heightInt * heightResolution + heightResolution : -heightInt * heightResolution_Underground;
		
        latMin = latMin < -90 ? -90 : latMin;
		lonMin = lonMin < -180 ? -180 : lonMin;
        latMax = latMax > 90 ? 90 : latMax;
        lonMax = lonMax > 180 ? 180: lonMax;
		
        return {
            level: level,
            min_coord: [latMin, lonMin, heightMin],
            max_coord: [latMax, lonMax, heightMax]
        }
    }

	


function get_neighbours_2d(code){
	console.assert(code.length % 2 === 0 && code.length > 2);
	let level = code.length / 2;
	let code_origin_lat = '';
	let code_origin_lon = '';
	//注意这块儿是从0开始的，不是从3开始的
	for (var i = 0; i < code.length; i++) {
		if(i%2==0){
		code_origin_lat = code_origin_lat.concat(code[i]);
		}
		if(i%2==1){
		code_origin_lon = code_origin_lon.concat(code[i]);
		}
	}
	let code_origin_lat_Int = parseInt(code_origin_lat, 2);
	let code_origin_lon_Int = parseInt(code_origin_lon, 2);
	let code_north_lat_Int = code_origin_lat_Int+1;
	let code_south_lat_Int = code_origin_lat_Int-1;
	let code_east_lon_Int = code_origin_lon_Int+1;
	let code_west_lon_Int = code_origin_lon_Int-1;
	let code_north_lat = code_north_lat_Int.toString(2);
	code_north_lat = Array(level +1 - code_north_lat.length).join('0') + code_north_lat;
	let code_south_lat = code_south_lat_Int.toString(2);
	code_south_lat = Array(level +1 - code_south_lat.length).join('0') + code_south_lat;
	let code_east_lon = code_east_lon_Int.toString(2);
	code_east_lon = Array(level +1 - code_east_lon.length).join('0') + code_east_lon;
	let code_west_lon = code_west_lon_Int.toString(2);
	code_west_lon = Array(level +1 - code_west_lon.length).join('0') + code_west_lon;

	console.log(code_origin_lat.length);
	console.log(code_origin_lon.length);
	console.log(code_north_lat.length);
	console.log(code_south_lat.length);
	console.log(code_east_lon.length);
	console.log(code_west_lon.length);

 	if (code_south_lat_Int < 0){
		code_south_lat = code_origin_lat;
		}

 	if (code_west_lon_Int < 0){
		code_west_lon = code_origin_lon;
		}
	let code_north = combinecode_2_1(code_north_lat,code_origin_lon);
	let code_south = combinecode_2_1(code_south_lat,code_origin_lon);
	let code_east = combinecode_2_1(code_origin_lat,code_east_lon);
	let code_west = combinecode_2_1(code_origin_lat,code_west_lon);
	let code_northwest = combinecode_2_1(code_north_lat,code_west_lon);
	let code_southhwest = combinecode_2_1(code_south_lat,code_west_lon);
	let code_northeast = combinecode_2_1(code_north_lat,code_east_lon);
	let code_southheast = combinecode_2_1(code_south_lat,code_east_lon);
	return [code,code_north,code_south,code_east,code_west,code_northwest,code_southhwest,code_northeast,code_southheast];
}

function combinecode_2_1(code_lat,code_lon){
	console.assert(code_lat.length === code_lon.length);
	result_code = '';
	var len = code_lat.length;
	for(var i = 0; i < len; i++){
		result_code = result_code.concat(code_lat[i]);
		result_code = result_code.concat(code_lon[i]);
	}
	return result_code;
}		

function combinecode_2_1(code_lat,code_lon){
	console.assert(code_lat.length === code_lon.length);
	result_code = '';
	var len = code_lat.length;
	for(var i = 0; i < len; i++){
		result_code = result_code.concat(code_lat[i]);
		result_code = result_code.concat(code_lon[i]);
	}
	return result_code;
}		




	''')   #javascript文件


def geosot_neighbor2st_fun(geosot_code):
	geosot_neighbor=ctx.call("get_neighbours_2d", geosot_code)

	geosot_neighbor2st=[]
	for i in range(len(geosot_neighbor)):
		geosot_neighbor2st.extend(ctx.call("get_neighbours_2d", geosot_neighbor[i]))
	# lst = list(set(lst))
	geosot_neighbor2st = list(OrderedDict.fromkeys(geosot_neighbor2st))
	return(geosot_neighbor2st)

def sort_and_rank(score, target):   #按照指定的维度对输入张量的元素进行排序，返回排序后的张量和对应的索引。
    #score 是一个包含分数的张量，target 是一个包含目标值的张量。
    _, indices = torch.sort(score, dim=1, descending=True) #第一行代码按照第 1 维度将分数按降序排序，并返回排序后的分数和它们对应的索引
    # print('score:',score,score.shape)
    # print('indices:',indices,indices.shape)
    indices = torch.nonzero(indices == target.view(-1, 1)) #在排序后的分数张量中找到目标值的索引
    # print('target:',target,target.shape)
    # print('indices11:',indices,indices.shape)
    indices = indices[:, 1].view(-1)     # 从索引张量中提取目标值的索引，并将它们作为一个新张量命名为 indices返回。
    # print('indices22:',indices,indices.shape)
    return indices   #indices每一个的排名

def sort_and_rank_grid(score, target, intention_list):
    cfg = _get_grid_cfg()
    if cfg is None:
        # fallback（不认识的 dataset 就用 ICEWS14s 的默认）
        grid_start_id = 7617
        filename = "locgrid_one2one_level7-14s.csv"
    else:
        grid_start_id = cfg["grid_start_id"]
        filename = cfg["locgrid_csv"]

    csv_path = _resolve_csv_path(filename)
    locgrid_one2one_level = _get_locgrid_df(csv_path)

    _, indices = torch.sort(score, dim=1, descending=True)

    new_tensor = []
    intention = indices.tolist()
    intention_list.append(intention)

    for i in range(indices.shape[0]):
        tgt = target.view(-1, 1)[i].item()

        if tgt >= grid_start_id:
            row = locgrid_one2one_level.loc[locgrid_one2one_level['id'] == tgt]
            geosot = "%014d" % row.iloc[0]['geosot']
            geosot_neighbor2st = geosot_neighbor2st_fun(geosot)
            geosot_id = []
            for k in range(len(geosot_neighbor2st)):
                row2 = locgrid_one2one_level.loc[locgrid_one2one_level['geosot'] == int(geosot_neighbor2st[k])]
                if not row2.empty:
                    geosot_id.append(row2.iloc[0]['id'])

        for j in range(indices.shape[1]):
            if tgt < grid_start_id:
                if indices[i][j].item() == tgt:
                    new_tensor.append([i, j])
                    break
            else:
                if indices[i][j].item() in geosot_id:
                    new_tensor.append([i, j])
                    break

    indices = torch.tensor(new_tensor)
    indices = indices[:, 1].view(-1)
    return indices
            
    # print('score:',score,score.shape)
    # indices = torch.nonzero((indices >= target.view(-1, 1)-20) & (indices <= target.view(-1, 1)+20)) #在排序后的分数张量中找到目标值的索引
    # indices = new_tensor.tolist()
    indices = torch.tensor(new_tensor)
    # appeared = set()
    # # Then, create a list to store the rows that have unique elements in the first column
    # new_tensor = []
    # for row in indices:
    #     if row[0] not in appeared:
    #         appeared.add(row[0])
    #         new_tensor.append(row)
    #     else:
    #         for i in range(len(new_tensor)):
    #             if new_tensor[i][0] == row[0]:
    #                 break
    #         else:
    #             new_tensor.append(row)
    # indices = torch.tensor(new_tensor)
    # print(indices)
    indices = indices[:, 1].view(-1)     # 从索引张量中提取目标值的索引，并将它们作为一个新张量命名为 indices返回。
    # print(indices)
    return indices   #indices每一个的排名


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2] 第一列递增， 第二列表示对应的答案实体id在每一行的位置
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    Build DGL graph for one snapshot.
    Optimization: do degree/norm on CPU first; move to GPU at the end.
    """
    def comp_deg_norm_cpu(g_cpu):
        in_deg = g_cpu.in_degrees().float()  # CPU
        in_deg[in_deg == 0] = 1
        return 1.0 / in_deg

    src, rel, dst = triples.transpose()
    src_all = np.concatenate((src, dst))
    dst_all = np.concatenate((dst, src))
    rel_all = np.concatenate((rel, rel + num_rels))

    # 1) build on CPU
    g = dgl.graph((src_all, dst_all), num_nodes=num_nodes)

    # 2) node features on CPU
    norm = comp_deg_norm_cpu(g).view(-1, 1)                 # CPU
    node_id = torch.arange(num_nodes, dtype=torch.long).view(-1, 1)  # CPU
    g.ndata.update({'id': node_id, 'norm': norm})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})

    # 3) edge features on CPU
    g.edata['type'] = torch.LongTensor(rel_all)

    # 4) meta attrs
    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_len = r_len
    g.r_to_e = torch.from_numpy(np.array(r_to_e))  # CPU tensor

    # 5) move to GPU at end
    if use_cuda:
        device = torch.device(f"cuda:{gpu}") if isinstance(gpu, int) else torch.device(gpu)
        g = g.to(device)
        g.r_to_e = g.r_to_e.to(device, non_blocking=True)

    return g

from collections import OrderedDict

# CPU 图缓存（LRU）
subgraph_cache_cpu = OrderedDict()
# GPU 图缓存（LRU，建议很小）
subgraph_cache_gpu = OrderedDict()

def _move_graph_attrs_to(g_src, g_dst, device):
    """把 g_src 的图属性同步到 g_dst，并把 tensor 属性搬到 device"""
    g_dst.uniq_r = g_src.uniq_r
    g_dst.r_len = g_src.r_len
    if hasattr(g_src, "r_to_e"):
        if isinstance(g_src.r_to_e, torch.Tensor):
            g_dst.r_to_e = g_src.r_to_e.to(device, non_blocking=True)
        else:
            g_dst.r_to_e = torch.tensor(g_src.r_to_e, device=device)
    return g_dst

def get_subgraph_cached(triples, num_nodes, num_rels, use_cuda, gpu,
                        cpu_cache_size=200000, gpu_cache_size=128):
    """
    关键点：
    1) CPU 缓存：保存 build_sub_graph(..., use_cuda=False)
    2) GPU 缓存：只缓存最近用到的一小部分，防止显存爆
    """
    key = hash(triples.tobytes())

    # ---------- GPU cache ----------
    if use_cuda and key in subgraph_cache_gpu:
        subgraph_cache_gpu.move_to_end(key)
        return subgraph_cache_gpu[key]

    # ---------- CPU cache ----------
    if key in subgraph_cache_cpu:
        subgraph_cache_cpu.move_to_end(key)
        g_cpu = subgraph_cache_cpu[key]
    else:
        # 永远在 CPU 构建并缓存
        g_cpu = build_sub_graph(num_nodes, num_rels, triples, use_cuda=False, gpu=gpu)
        subgraph_cache_cpu[key] = g_cpu
        subgraph_cache_cpu.move_to_end(key)
        if len(subgraph_cache_cpu) > cpu_cache_size:
            subgraph_cache_cpu.popitem(last=False)

    # ---------- move to GPU if needed ----------
    if use_cuda:
        device = torch.device(f"cuda:{gpu}") if isinstance(gpu, int) else torch.device(gpu)
        g_gpu = g_cpu.to(device)
        g_gpu = _move_graph_attrs_to(g_cpu, g_gpu, device)

        subgraph_cache_gpu[key] = g_gpu
        subgraph_cache_gpu.move_to_end(key)
        if len(subgraph_cache_gpu) > gpu_cache_size:
            subgraph_cache_gpu.popitem(last=False)

        return g_gpu

    return g_cpu

def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)  #将列表rank_list中的所有张量沿着默认维度0连接起来，并将结果赋值给一个名为total_rank的新张量。
    # print('totoal_rank:',total_rank)  #  totoal_rank: tensor([11,  8,  1,  ...,  1,  7,  1], device='cuda:0')
    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr
def get_total_rank(test_triples, score, all_ans, eval_bz=1000, rel_predict=0):
    """
    统一计算 raw / filtered 的排名与 MRR/Hits

    参数：
      test_triples: (N, 3) LongTensor，形如 [h, r, t]
      score:        (N, num_ents) 或 (N, 2*num_rels) 的打分矩阵
      all_ans:      load_all_answers_for_time_filter 得到的 all_ans_list[time_idx]
      eval_bz:      每次评估的 batch 大小
      rel_predict:  0 = 实体预测（目标是 t），1 = 关系预测（目标是 r）

    返回：
      mrr_filter_snap: filtered MRR（当前时间片）
      mrr_snap:        raw MRR（当前时间片）
      rank_raw:        所有样本的 raw rank（1-based）
      rank_filter:     所有样本的 filtered rank（1-based）
    """
    device = score.device
    num_examples = test_triples.size(0)

    ranks_raw_list = []
    ranks_filter_list = []

    for start in range(0, num_examples, eval_bz):
        end = min(start + eval_bz, num_examples)

        batch_triples = test_triples[start:end].to(device)
        batch_score_raw = score[start:end].clone()      # 用于 raw rank
        batch_score_flt = score[start:end].clone()      # 用于 filtered rank

        if rel_predict == 0:
            # ========= 实体预测：目标是 tail =========
            batch_h = batch_triples[:, 0]
            batch_r = batch_triples[:, 1]
            target  = batch_triples[:, 2]

            # raw: 不做过滤
            ranks_raw = sort_and_rank(batch_score_raw, target)

            # filtered: 屏蔽其它正确 tail
            batch_score_flt = filter_score(batch_triples, batch_score_flt, all_ans)
            ranks_filter = sort_and_rank(batch_score_flt, target)

            method_raw = "raw-entity"
            method_flt = "filt-entity"

        else:
            # ========= 关系预测：目标是 relation =========
            batch_h = batch_triples[:, 0]
            batch_t = batch_triples[:, 2]
            target  = batch_triples[:, 1]

            # raw: 不做过滤
            ranks_raw = sort_and_rank(batch_score_raw, target)

            # filtered: 屏蔽其它正确 relation
            batch_score_flt = filter_score_r(batch_triples, batch_score_flt, all_ans)
            ranks_filter = sort_and_rank(batch_score_flt, target)

            method_raw = "raw-relation"
            method_flt = "filt-relation"

        # sort_and_rank 返回的是 0-based 排名，这里统一转成 1-based
        ranks_raw_list.append(ranks_raw + 1)
        ranks_filter_list.append(ranks_filter + 1)

    # 拼接所有 batch 的 rank
    rank_raw = torch.cat(ranks_raw_list)
    rank_filter = torch.cat(ranks_filter_list)

    # 统计当前 snapshot 的 MRR/Hits（会打印详细信息）
    mrr_snap = stat_ranks([rank_raw], method_raw)
    mrr_filter_snap = stat_ranks([rank_filter], method_flt)

    return mrr_filter_snap, mrr_snap, rank_raw, rank_filter

def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):   
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)

def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel): #如果字典中不存在键e1，则添加一个新的键e1，并将其值设置为空字典{}。如果字典中存在键e1，则不做任何操作。接下来，如果字典d[e1]中不存在键r，则添加一个新的键r，并将其值设置为空集合set()。如果字典d[e1]中存在键r，则不做任何操作。最后，将元素e2添加到集合d[e1][r]中。1
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)

def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)
    
    # print(all_ans_list[2][3][9])  #[时间切片][主语或宾语][主语宾语或关系]={主语宾语或关系}    如果添加关系  d[e1][e2].add(r)
    # output_label_list = []
    # for all_ans in all_ans_list:
    #     output = []
    #     ans = []
    #     for e1 in all_ans.keys():
    #         for r in all_ans[e1].keys():
    #             output.append([e1, r])
    #             ans.append(list(all_ans[e1][r]))
    #     output = torch.from_numpy(np.array(output))
    #     output_label_list.append((output, ans))
    # return output_label_list
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    # 处理空快照列表的情况
    if len(snapshot_list) == 0:
        print("# Sanity Check:  [Warning] No snapshots created! Data might be empty or all filtered out.")
        return snapshot_list

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k  # k=1 需要取长度k的历史，在加1长度的label
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

    def load_from_local(data_dir, dataset):
        data = knwlgrh.RGCNLinkDataset(dataset, dir=data_dir)
        data.load()
        return data

    # NEW: 记录当前 dataset，供 sort_and_rank_grid 自动切换
    _set_current_dataset(dataset)

    if dataset in ['ICEWS18', 'ICEWS14', 'ICEWS14s', 'ICEWS05-15','YAGO', 'WIKI', 'GDELT', 'OPENBIOLINK', 'FB15k-237']:
        return load_from_local(DATA_DIR, dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a
