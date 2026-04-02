import cv2
import pandas as pd
import numpy as np


from PIL import Image
import streamlit as st
from streamlit import session_state as state
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_drawable_canvas import st_canvas
np.seterr(divide = "ignore",invalid="ignore")

st.set_page_config(layout="wide")



tracked_vars = ["centre_x","centre_y","width","height","tl_x","tl_y","br_x","br_y","draw_colour","draw_colour_hex","scale"]
default_vals = [200,200,100,100,150,150,250,250,"Red","FF0000",5]

for (var_i, val_i) in zip(tracked_vars, default_vals):
    if var_i not in state:
        state[var_i] = val_i


temp_col1,temp_col2= st.columns(2)

with temp_col1:
    raw_img = st.file_uploader("Upload An Image",type = ["jpg","jpeg","png"])
if raw_img is not None:
    with temp_col2:
        state["scale"] = st.slider(r"$\textsf{\Large Display Scale Factor of Subsection}$",min_value=0.5,max_value=10.0,value = 5.0,step = 0.01)
    
    img_pil = Image.open(raw_img)
    img_color = np.array(img_pil).reshape(img_pil.size[1],img_pil.size[0],3)
    col1,col2 = st.columns(2)    



    coord_img_size_x = 625

    coord_img_ratio = img_color.shape[1]/coord_img_size_x

    img_sub = img_color.copy()[state["tl_y"]:state["br_y"],state["tl_x"]:state["br_x"]] # must reset every page otherwise 


    def correct_centre():

        if 0.5*state["width"] + state["centre_x"] > img_color.shape[1]: 
            state["tl_x"] = img_color.shape[1]-state["width"]

        elif state["centre_x"] - 0.5*state["width"] < 0:
            state["tl_x"] =  0

        else:
            state["tl_x"] = int(state["centre_x"] - 0.5*state["width"])

        if 0.5*state["height"] + state["centre_y"] > img_color.shape[0]:
            state["tl_y"] = img_color.shape[0] - state["height"]

        elif state["centre_y"] - 0.5*state["height"] < 0:
            state["tl_y"] =  0

        else:
            state["tl_y"] = int(state["centre_y"] - 0.5*state["height"])
        state["br_x"] = state["tl_x"]+state["width"]
        state["br_y"] = state["tl_y"]+state["height"]

    ############################## main loop ##########################################################################################################################################################



    def join_points(contour_list,pt): # automatically uses the last point of the contour list as the first point
        if len(contour_list) == 0:
            contour_list.append(pt)
            return contour_list, False
        
        current_pt = np.copy(contour_list[-1])
        pt = np.copy(pt)
        x_vec = pt[0]-current_pt[0]
        y_vec = pt[1]-current_pt[1]
        
        
        larger_component = max(abs(x_vec),abs(y_vec))
        vec_normalised=np.divide((pt-current_pt),larger_component)

        for j in range(int(larger_component)): # to be an even line between the two points
            current_pt += vec_normalised
            if np.array_equal(current_pt, contour_list[0]) and len(contour_list) > 1:
                return contour_list,True
            contour_list.append(np.round(current_pt))
            
        return contour_list,False


    with col1:

        state["width"] = st.slider(r"$\textsf{\Large Width of Subsection (pixels)}$",  min_value=4,max_value=img_color.shape[1]//2,step = 2, value=100,)
        st.write("Click the Image below to get a subsecttion around where you clicked. You can change the size of this using the sliders above!")

        centre = streamlit_image_coordinates(img_color, width = coord_img_size_x, cursor="crosshair")
        if centre == None:
            centre = {"x":state["centre_x"]/coord_img_ratio, "y":state["centre_y"]/coord_img_ratio}

        state["centre_x"] = coord_img_ratio*centre["x"]
        state["centre_y"] = coord_img_ratio*centre["y"]
        correct_centre()
        
        
        

    with col2:
        loop_closed = False

        state["height"] = st.slider(r"$\textsf{\Large Height of Subsection (pixels)}$",min_value=4,max_value=img_color.shape[0]//2,step = 2, value=100)
        st.write("Your subsection should appear here. If it does not appear correctly, try clicking on the left image again!")

        img_sub = img_color.copy()[state["tl_y"]:state["br_y"],state["tl_x"]:state["br_x"]]
        cv2.imwrite("temp_image.png",img_sub)

        state["draw_colour"] = st.selectbox("Draw Colour",["Red", "Green", "Blue", "Purple"])
        colour_hex_dict = {"Red":"#FF0000", "Green":"#00FF00", "Blue":"#0000FF", "Purple":"#9900FF"}
        state["draw_colour_hex"] = colour_hex_dict[state["draw_colour"]]

        canvas_result = st_canvas(
            background_image=Image.fromarray(img_sub),
            stroke_color=state["draw_colour_hex"],
            stroke_width=5,
            width=img_sub.shape[1]*state["scale"],
            height=img_sub.shape[0]*state["scale"]
            )

        
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
                        path_df = pd.concat([pd.DataFrame([starts_list[k][1:]]),path_df,pd.DataFrame([ends_list[k][1:]])]).apply(pd.to_numeric).rename(columns = {0:"x",1:"y"}).apply(np.round).reset_index(drop=True)

                    # duplicates should be allowed, but not one after the other
                    
                    path_df_displaced = pd.concat([path_df.loc[0:0]+1,path_df[:-1]])
                    path_df["dupe_condition"] = (path_df_displaced.reset_index(drop=True)-path_df).apply(lambda p:p**2).sum(1)
                    path_df=path_df[path_df["dupe_condition"]!=0].reset_index(drop=True).drop(columns="dupe_condition")
                    
                    
                    final_path_arr = np.array(path_df)
                    
    ################################### contour logic/completing ###############################################################################################################################################################

                    
                    if len(contour_list) != 0:
                        contour_list,loop_closed = join_points(contour_list,final_path_arr[0])
                        if loop_closed:
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
                if not loop_closed:
                    contour_list,loop_closed = join_points(contour_list,contour_list[0])
                    loop_closed = True
                    
        
        if st.button("Export Contour") and loop_closed:
            export_array = np.array(contour_list)/state["scale"] + np.array([state.tl_x, state.tl_y])
            export_df: pd.DataFrame = pd.DataFrame(np.round(export_array)).drop_duplicates().reset_index(drop = True)

            mode = "w"
            try:
                current_file = pd.read_csv("contour.csv")
                
                if len(current_file) == 0:
                    contour_id = 0
                    header = ["x","y","contour_id","colour"]
                else:
                    contour_id = current_file.loc[current_file.index[-1],"contour_id"] + 1 
                    header = False
                    mode = "a"
                    
                f = open("contour.csv","a")
                f.write("\n")
                f.close()
                    
            except:
                contour_id = 0
                header = ["x","y","contour_id","colour"]
            
            
            export_df["contour_id"] = contour_id
            export_df["colour"] = objects.loc[0,"stroke"]
            export_df.to_csv("contour.csv",mode = mode,float_format=int,index=False,header=header)

    with col1:
        try:
            contour_df = pd.read_csv("contour.csv") 
            max_id = contour_df["contour_id"].max() 

            if len(contour_df["contour_id"].unique()) > 1:
                selected_contour = st.selectbox("Select a Contour", contour_df["contour_id"].unique())
                
            else:
                selected_contour = max_id

            if st.button("Delete Selected Contour (in Yellow)"):
                contour_df = pd.read_csv("contour.csv")
                contour_df = contour_df[contour_df["contour_id"]!=selected_contour]
                
                def reorder_contours(contour_df):
                    id_list = list(contour_df["contour_id"].unique())
                    ideal_mapping = list(range(len(id_list)))
                    map_dict = dict(zip(id_list,ideal_mapping))
                    contour_df["contour_id"] = contour_df["contour_id"].apply(lambda x: map_dict[x])
                    return contour_df
                
                contour_df = reorder_contours(contour_df)
                contour_df[contour_df["contour_id"]!=max_id].to_csv("contour.csv",index = False)
                

            img_overlay = img_color.copy()
            
            


            for i in list(contour_df["contour_id"].unique()):
                current_contour = contour_df[contour_df["contour_id"]==i]
                current_colour = current_contour["colour"].min()

                current_colour = (int(current_colour[1:3],16),int(current_colour[3:5],16),int(current_colour[5:7],16)) # converts hex code to denary tuple 

                cv2.drawContours(img_overlay, [np.array(current_contour.drop(columns=["contour_id","colour"]))], -1, current_colour, 2)
                    
            
            if len(contour_df) != 0:
                current_contour = contour_df[contour_df["contour_id"]==selected_contour]
                cv2.drawContours(img_overlay, [np.array(current_contour.drop(columns=["contour_id","colour"]))], -1, (255, 255, 0), 2) # yellow overrides all other colours

            st.image(img_overlay)
        except:
            pass

        


