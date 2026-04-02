import cv2
import pandas as pd
import numpy as np
 
 
from PIL import Image
import streamlit as st
from streamlit import session_state as state
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas

img = cv2.imread('Image4.png', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('Image4.png', cv2.IMREAD_COLOR_RGB)
st.set_page_config(layout="wide")

IMG_SIZE_X = 1100
IMG_SIZE_Y = 530 # based on img size Y of 1100

PIXEL_RATIO = img.shape[1]/IMG_SIZE_X


tracked_vars = ["centre_x","centre_y","width","height","tl_x","tl_y","br_x","br_y","draw_colour"]
default_vals = [200,200,100,100,150,150,250,250,"FF0000"]

for (var_i, val_i) in zip(tracked_vars, default_vals):
    if var_i not in state:
        state[var_i] = val_i

img_sub = img_color.copy()[state["tl_y"]:state["br_y"],state["tl_x"]:state["br_x"]] # must reset every page otherwise 
cv2.imwrite("temp_images/temp_image.png",img_sub,)

# rect_img = img_color.copy()
# cv2.rectangle(rect_img,(state["tl_x"],state["tl_x"]),(state["tl_x"]+state["width"],state["tl_y"]+state["height"]), (255,255,255), 2)



def correct_centre():

    if 0.5*state["width"] + state["centre_x"] > img.shape[1]: 
        state["tl_x"] = img.shape[1]-state["width"]

    elif state["centre_x"] - 0.5*state["width"] < 0:
        state["tl_x"] =  0

    else:
        state["tl_x"] = int(state["centre_x"] - 0.5*state["width"])

    if 0.5*state["height"] + state["centre_y"] > img.shape[0]:
        state["tl_y"] = img.shape[0] - state["height"]

    elif state["centre_y"] - 0.5*state["height"] < 0:
        state["tl_y"] =  0

    else:
        state["tl_y"] = int(state["centre_y"] - 0.5*state["height"])
    state["br_x"] = state["tl_x"]+state["width"]
    state["br_y"] = state["tl_y"]+state["height"]

############################## main loop ##########################################################################################################################################################

col1, col2 = st.columns(2)

def join_points(contour_list,pt): # automatically uses the last point of the contour list as the first point
    if len(contour_list) == 0:
        contour_list.append(pt)
        return contour_list, False
    
    current_pt = np.copy(contour_list[-1])
    pt = np.copy(pt)
    x_vec = pt[0]-current_pt[0]
    y_vec = pt[1]-current_pt[1]
    
    
    larger_component = max(abs(x_vec),abs(y_vec))
    vec_normalised=(pt-current_pt)/larger_component

    for j in range(int(larger_component)): # to be an even line between the two points
        current_pt += vec_normalised
        if np.array_equal(current_pt, contour_list[0]) and len(contour_list) > 1:
            st.write("aaa")
            return contour_list,True
        contour_list.append(np.round(current_pt))
        
    return contour_list,False

with col1:
    

    state["width"] = st.slider(r"$\textsf{\Large Width of Subsection (pixels)}$",  min_value=4,max_value=img.shape[1]//2,step = 2, value=100)
    
    centre = streamlit_image_coordinates(img_color, width = IMG_SIZE_X, cursor="crosshair")
    if centre == None:
        centre = {"x":state["centre_x"]/PIXEL_RATIO, "y":state["centre_y"]/PIXEL_RATIO}
    state["centre_x"] = PIXEL_RATIO*centre["x"]
    state["centre_y"] = PIXEL_RATIO*centre["y"]
    correct_centre()
    
    

with col2:
    stop = -1 # parameter that will be used later
    loop_closed = False
    SCALE = 5
    state["height"] = st.slider(r"$\textsf{\Large Height of Subsection (pixels)}$",min_value=4,max_value=img.shape[0]//2,step = 2, value=100)

    img_sub = img_color.copy()[state["tl_y"]:state["br_y"],state["tl_x"]:state["br_x"]]
    cv2.imwrite("temp_images/temp_image.png",img_sub,)
    canvas_result = st_canvas(
        background_image=Image.open("temp_images/temp_image.png"),
        stroke_color=state["draw_colour"],
        stroke_width=5,
        width=img_sub.shape[1]*SCALE,
        height=img_sub.shape[0]*SCALE
        )

    state["draw_colour"] = st.color_picker("Draw Colour:",value="#FF0000")
################################################# Data Handling - canvas paths #################################################################################################################################################

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        contour_list = []

        if len(objects)!= 0:
            path_list = objects["path"].tolist()
            starts_list = []
            ends_list = []
            for path_i in path_list:
                starts_list.append(path_i.pop(0))
                ends_list.append(path_i.pop(-1))
            
            stop = -1
            loop_closed= False

            for k in range(len(path_list)):
                

                path_arr= np.array(path_list[k])
                if len(path_list[k]) == 0:
                    path_df = pd.concat([pd.DataFrame([starts_list[k][1:]]),pd.DataFrame([ends_list[k][1:]])]).apply(pd.to_numeric).rename(columns = {0:"x",1:"y"}).apply(np.round).reset_index(drop=True)
                else:
                    path_df = pd.DataFrame(path_arr,columns=["type","x1","y1","x2","y2"]).drop(columns = "type")
                    
                    # by default each row contains 2 points, must use stack to order these correctly
                    path_df = pd.concat([path_df[["x1","x2"]].stack().reset_index(drop=True),path_df[["y1","y2"]].stack().reset_index(drop=True)],axis=1)
                    path_df = pd.concat([pd.DataFrame([starts_list[-1][1:]]),path_df,pd.DataFrame([ends_list[-1][1:]])]).apply(pd.to_numeric).rename(columns = {0:"x",1:"y"}).apply(np.round).reset_index(drop=True)

                # duplicates should be allowed, but not one after the other
                path_df_displaced = pd.concat([path_df.loc[0:0]+1,path_df[:-1]])
                path_df["dupe_condition"] = (path_df_displaced.reset_index(drop=True)-path_df).apply(lambda p:p**2).sum(1)
                path_df=path_df[path_df["dupe_condition"]!=0].reset_index(drop=True).drop(columns="dupe_condition")
                
                
                final_path_arr = np.array(path_df)
                st.write(final_path_arr)
################################### contour logic/completing ###############################################################################################################################################################

                
                if len(contour_list) != 0:
                    if np.linalg.norm(contour_list[-1]-final_path_arr[0]) <= 15: # if new path start is close to old path end
                        contour_list,loop_closed = join_points(contour_list,final_path_arr[0])
                        if loop_closed:
                            break
                    else:
                        st.write(contour_list)
                        st.write(final_path_arr)
                        stop = k+1
                        break

                if len(final_path_arr) == 1:
                    contour_list.append(final_path_arr[0])
                
                else:

                    for i in range(len(final_path_arr)):
                        contour_list,loop_closed = join_points(contour_list,final_path_arr[i])
                        if loop_closed:
                            break
                    if loop_closed:
                        break
                
            if st.checkbox("Finish My Contour"): # tie up the end if they have not already
                if not loop_closed:
                    contour_list,loop_closed = join_points(contour_list,contour_list[0])
                    loop_closed = True

            if stop != -1:
                st.write(f"Line No. {k+1} was not close enough to Line {k}, please undo this")
    if loop_closed:
        st.write(np.array(contour_list))
            


